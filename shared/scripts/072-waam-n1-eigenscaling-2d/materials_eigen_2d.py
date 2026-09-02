#!/usr/bin/env python3
"""
Material assignment for project 072 (eigen-scaled transition stiffness) —
shared by BOTH solvers (`solve_plane_stress_eigen.py`, dolfinx;
`reference_solver_numpy.py`), so the two differ only in the FE machinery,
never in the material law.

Same assignment rule as 070 (region -> material -> cubic constants per
crystal system, per-grain Bond rotation, exact plane-stress condensation),
but the scalar transition-zone prefactor s(x) is generalised to THREE
independent, spatially varying factors on the irreducible parts of the cubic
tensor (crystal frame, before rotation):

    K   = (C11 + 2 C12)/3   bulk           -> factor aK(x)
    C'  = (C11 - C12)/2     tetragonal shear -> factor aCp(x)
    C44                     trigonal shear   -> factor aC44(x)

    C_cell = P( R(g_grain) . [ aK*Ch + aCp*Ct + aC44*Cs ] )

with Ch+Ct+Cs the eigen-parts of C_cubic[material][crystal system]
(plane_stress_crystal.cubic_eigen_parts), R the Bond rotation by that grain's
own Bunge angles, P the exact plane-stress condensation. The factors are
applied ONLY in region SCALED_REGION (the transition zone, as in 070).

aK = aCp = aC44 = s reproduces 070's scalar s(x) exactly (rotation and
condensation are linear resp. degree-1 homogeneous in C). The Zener ratio of
the scaled crystal is A_scaled = (aC44/aCp) * A, so aCp and aC44 change the
ANISOTROPY, not just the magnitude. Positive factors always keep C positive
definite (the three parts are the eigenspaces of the cubic tensor).

All constants live in `config.json` next to this file — that is the single
place to edit them (same format and values as 070).
"""
import json
import os

import numpy as np

from plane_stress_crystal import (bunge_to_g, rotate_C, plane_stress_condense,
                                  cubic_C, cubic_eigen_parts, cubic_to_KCpC44,
                                  FLIP_X180)

# Literature PLACEHOLDERS [GPa] — same values as 069/070 config.json.
DEFAULT_CUBIC = {
    "17-4PH": {                                   # martensitic, region 0
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
    },
    "316L": {                                     # austenitic, region 2
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},
    },
    "trans": {                                    # transition zone, region 1
        "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
        "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},
    },
}
REGION_MATERIAL = {0: "17-4PH", 1: "trans", 2: "316L"}
PHASE_CRYSTAL = {1: "fcc", 2: "bcc"}
SCALED_REGION = 1                     # only the transition zone gets factors
FACTOR_NAMES = ("aK", "aCp", "aC44")


def load_config(path=None, here=None):
    """Read config.json; missing entries fall back to DEFAULT_CUBIC.
    Identical logic/format as 070/materials_2d.load_config."""
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
            if not isinstance(const, dict) or "C11" not in const:
                continue
            if crystal in per_crystal:
                per_crystal[crystal].update(const)
            else:
                per_crystal[crystal] = dict(const)
    if "region_material" in user:
        cfg["region_material"] = {int(k): v for k, v in user["region_material"].items()}
    return cfg


def make_factor_funs(aK="1.0", aCp="1.0", aC44="1.0", sfun=None):
    """Three callables f(x_um) from numpy-expression strings. If `sfun` is
    given it overrides all three (070 compatibility: aK=aCp=aC44=s)."""
    if sfun is not None:
        aK = aCp = aC44 = sfun
    return {name: eval('lambda x: ' + expr, {'np': np})   # noqa: S307
            for name, expr in zip(FACTOR_NAMES, (aK, aCp, aC44))}


def E_directional(C2, theta_deg=0.0):
    """Young's modulus of a 3x3 plane-stress stiffness in direction theta
    (deg from x). E = 1 / S'_11 with S' the rotated compliance."""
    S = np.linalg.inv(C2)
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    s11 = (S[0, 0] * c**4 + S[1, 1] * s**4
           + (2 * S[0, 1] + S[2, 2]) * c**2 * s**2
           + 2 * (S[0, 2] * c**2 + S[1, 2] * s**2) * c * s)
    return 1.0 / s11


def build_cell_tensors(euler, phase, gid, zone, x_um, cfg, afuns=None,
                       verbose=True):
    """Per-cell plane-stress stiffness for the whole microstructure grid.

    Same signature/semantics as 070/materials_2d.build_cell_tensors, but the
    scalar sfun is replaced by `afuns` = {"aK": f, "aCp": f, "aC44": f}
    (from make_factor_funs; default all 1.0), applied in SCALED_REGION only.

    Returns
    -------
    C     : (ny,nx,3,3) stiffness [GPa]
    Ex    : (ny,nx)     directional Young's modulus along x of each cell
    a_maps: {"aK","aCp","aC44"} -> (ny,nx) applied factors (1 elsewhere)
    info  : dict with grain counts and the constants actually used
    """
    ny, nx = phase.shape
    rmat = cfg["region_material"]
    afuns = afuns or make_factor_funs()

    # eigen-parts of the unrotated cubic tensor per material/crystal
    parts_tab = {mat: {cry: cubic_eigen_parts(c["C11"], c["C12"], c["C44"])
                       for cry, c in per.items()}
                 for mat, per in cfg["single_crystal_cubic_GPa"].items()}

    # factor value per x-column (they only depend on x)
    acol = {n: np.array([float(np.asarray(afuns[n](x))) for x in x_um])
            for n in FACTOR_NAMES}

    C = np.empty((ny, nx, 3, 3))
    Ex = np.empty((ny, nx))
    a_maps = {n: np.ones((ny, nx)) for n in FACTOR_NAMES}
    rot_cache = {}     # grain key -> (Bh, Bt, Bs) rotated 6x6 eigen-parts
    unscaled = {}      # grain key -> (C2, Ex) for factors == 1
    scaled = {}        # (grain key, column) -> (C2, Ex) inside the zone
    counts = {}
    for j in range(ny):
        zrow = zone[j]
        for i in range(nx):
            z = int(zrow[i])
            mat = rmat.get(z, rmat.get(str(z), "316L"))
            cry = PHASE_CRYSTAL.get(int(phase[j, i]), "fcc")
            key = (int(gid[j, i]), mat, cry)
            B = rot_cache.get(key)
            if B is None:
                g = bunge_to_g(*np.deg2rad(euler[j, i]))
                tab = parts_tab[mat]
                p3 = tab[cry] if cry in tab else next(iter(tab.values()))
                B = tuple(rotate_C(p, g, pre_rot=FLIP_X180) for p in p3)
                rot_cache[key] = B
                counts[(mat, cry)] = counts.get((mat, cry), 0) + 1
            if z == SCALED_REGION:
                hit = scaled.get((key, i))
                if hit is None:
                    C6 = (acol["aK"][i] * B[0] + acol["aCp"][i] * B[1]
                          + acol["aC44"][i] * B[2])
                    C2 = plane_stress_condense(C6)
                    hit = (C2, E_directional(C2))
                    scaled[(key, i)] = hit
                for n in FACTOR_NAMES:
                    a_maps[n][j, i] = acol[n][i]
            else:
                hit = unscaled.get(key)
                if hit is None:
                    C2 = plane_stress_condense(B[0] + B[1] + B[2])
                    hit = (C2, E_directional(C2))
                    unscaled[key] = hit
            C[j, i] = hit[0]
            Ex[j, i] = hit[1]

    info = {"n_distinct_grain_tensors": len(rot_cache),
            "grains_per_material_crystal": {f"{m}/{c}": n
                                            for (m, c), n in sorted(counts.items())},
            "region_material": {str(k): v for k, v in rmat.items()},
            "single_crystal_cubic_GPa": cfg["single_crystal_cubic_GPa"],
            "E_x_range_GPa": [float(Ex.min()), float(Ex.max())]}
    if verbose:
        print(f'Materialzuordnung: {info["grains_per_material_crystal"]}')
        print(f'  {len(rot_cache)} verschiedene Korn-Orientierungen; '
              f'E_x pro Zelle {Ex.min():.0f}...{Ex.max():.0f} GPa')
    return C, Ex, a_maps, info


def describe(cfg):
    """Human-readable dump of the constants in use, incl. K/C'/C44 split."""
    lines = ['Einkristall-Konstanten [GPa] (aus config.json, LITERATUR-PLATZHALTER):']
    for z, mat in sorted(cfg["region_material"].items()):
        for cry, c in cfg["single_crystal_cubic_GPa"][mat].items():
            K, Cp, C44 = cubic_to_KCpC44(c["C11"], c["C12"], c["C44"])
            lines.append(f'  region {z} = {mat:8s} {cry}: '
                         f'C11={c["C11"]:.1f} C12={c["C12"]:.1f} C44={c["C44"]:.1f}'
                         f'  ->  K={K:.1f} C\'={Cp:.1f} C44={C44:.1f}'
                         f'  (Zener A={C44 / Cp:.2f})')
    lines.append('Skalierung (nur region 1): C = P(R.[aK*Ch + aCp*Ct + aC44*Cs]);'
                 ' aK=aCp=aC44=s entspricht exakt 070.')
    return '\n'.join(lines)


if __name__ == '__main__':
    print(describe(load_config()))
