#!/usr/bin/env python3
"""
Material assignment for the 2D phase-field FRACTURE model (071).

ONE place defines what stiffness a point of the model gets:

    inside the microstructure patch   C(x) = s(x_map) * P( R(g_grain) . C_cubic[material][crystal] )
    outside (embedding)               C    = C_iso(E_emb, nu_emb)

with R the Bond rotation by that grain's own Bunge angles (map->FE flip
FLIP_X180, plus ROT_Z90 for the rotated transverse ROI), P the exact
plane-stress condensation, and s(x_map) the transition-zone prefactor of 070.

Gc is CONSTANT everywhere by design of this study (config.json -> fracture.
Gc_GPa_um). The ONLY heterogeneity is the elastic one. If that ever changes,
change it here and say so in the report.

Units: GPa and um throughout. 1 GPa*um = 1 kJ/m^2.

Runs with numpy only - no dolfinx - so it is testable outside the container
(`python3 materials_fracture_2d.py` prints the constants in use).
"""
import json
import os

import numpy as np

import crystal2d as X

# Literature PLACEHOLDERS [GPa] - same values as 069/070. Fallback only;
# config.json in this folder is the authoritative source.
DEFAULT_CUBIC = {
    "17-4PH": {"bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4},
               "fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2}},
    "316L":   {"fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
               "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4}},
    "trans":  {"fcc": {"C11": 204.6, "C12": 137.7, "C44": 126.2},
               "bcc": {"C11": 231.4, "C12": 134.7, "C44": 116.4}},
}
REGION_MATERIAL = {0: "17-4PH", 1: "trans", 2: "316L"}
PHASE_CRYSTAL = {1: "fcc", 2: "bcc"}
SCALED_REGION = 1                      # only the transition zone gets s(x)

DEFAULT_FRACTURE = {"Gc_GPa_um": 10.0, "epsilon_um": 12.0, "eta": 1.0e-4,
                    "mobility": 1000.0, "degradation": "quadratic", "crack_width_um": 24.0,
                    "degradation_beta": 0.1, "plane_state": "stress"}
DEFAULT_EMBEDDING = {"E_GPa": 200.0, "nu": 0.30,
                     "material": "isotropic steel (assumed)"}


# ---------------------------------------------------------------- config ---
def load_config(path=None, here=None):
    """Read config.json; missing entries fall back to the defaults above."""
    here = here or os.path.dirname(os.path.abspath(__file__))
    cfg = {"single_crystal_cubic_GPa": {m: {c: dict(v) for c, v in d.items()}
                                        for m, d in DEFAULT_CUBIC.items()},
           "region_material": dict(REGION_MATERIAL),
           "fracture": dict(DEFAULT_FRACTURE),
           "embedding": dict(DEFAULT_EMBEDDING),
           "roi": {}}
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
                continue          # skip "_comment"-style annotations
            per_crystal.setdefault(crystal, {}).update(const)
    if "region_material" in user:
        cfg["region_material"] = {int(k): v for k, v in user["region_material"].items()}
    for block in ("fracture", "embedding"):
        cfg[block].update({k: v for k, v in user.get(block, {}).items()
                           if not k.startswith("_")})
    cfg["roi"] = {k: v for k, v in user.get("roi", {}).items()
                  if not k.startswith("_")}
    return cfg


def constants_table(cfg):
    """{material: {crystal: 6x6 unrotated cubic C}} from a config dict."""
    return {mat: {cry: X.cubic_C(c["C11"], c["C12"], c["C44"])
                  for cry, c in per.items()}
            for mat, per in cfg["single_crystal_cubic_GPa"].items()}


def embedding_C2D(cfg):
    """3x3 stiffness of the homogeneous isotropic embedding [GPa]."""
    e = cfg["embedding"]
    return X.isotropic_C2D(e["E_GPa"], e["nu"], cfg["fracture"]["plane_state"])


def describe(cfg):
    """Human-readable dump of everything that fixes the material response."""
    f, e = cfg["fracture"], cfg["embedding"]
    lines = ["Einkristall-Konstanten [GPa] (config.json, LITERATUR-PLATZHALTER wie 069/070):"]
    for z, mat in sorted(cfg["region_material"].items()):
        for cry, c in cfg["single_crystal_cubic_GPa"][mat].items():
            lines.append(f'  region {z} = {mat:8s} {cry}: '
                         f'C11={c["C11"]:.1f} C12={c["C12"]:.1f} C44={c["C44"]:.1f}')
    lines.append(f'Einbettung (ANGENOMMEN, nicht homogenisiert): {e["material"]}, '
                 f'E={e["E_GPa"]:.1f} GPa, nu={e["nu"]:.3f}')
    deg = f["degradation"]
    if deg == "cubic":                       # beta only exists for the cubic form
        deg += " (beta={})".format(f["degradation_beta"])
    lines.append(f'Bruch: Gc={f["Gc_GPa_um"]:.4g} GPa*um (= kJ/m^2) KONSTANT ueberall, '
                 f'epsilon={f["epsilon_um"]:.4g} um, eta={f["eta"]:.1e}, '
                 f'Degradation={deg}, Zustand=plane {f["plane_state"]}')
    lines.append(f'  -> sig_c = {sig_c_estimate(cfg):.3f} GPa, '
                 f'K_eq = sqrt(E*Gc) = {K_from_Gc(f["Gc_GPa_um"], e["E_GPa"]):.1f} MPa*sqrt(m) '
                 f'(homogen-isotrope Abschaetzung)')
    return '\n'.join(lines)


def sig_c_estimate(cfg):
    """Critical stress of the 1D homogeneous phase-field model [GPa],
    evaluated with the embedding shear modulus - orientation only."""
    f, e = cfg["fracture"], cfg["embedding"]
    mu = e["E_GPa"] / (2.0 * (1.0 + e["nu"]))
    Gc, eps = f["Gc_GPa_um"], f["epsilon_um"]
    if f["degradation"] == "cubic":
        return 81.0 / 50.0 * np.sqrt(Gc * 2.0 * mu / (15.0 * eps))
    return 9.0 / 16.0 * np.sqrt(Gc * 2.0 * mu / (6.0 * eps))


# Unit conversion for stress intensity factors. In the GPa/um system K comes
# out in GPa*sqrt(um), and
#     1 GPa*sqrt(um) = 1e9 Pa * sqrt(1e-6 m) = 1e9 * 1e-3 Pa*sqrt(m)
#                    = 1e6 Pa*sqrt(m) = 1 MPa*sqrt(m)
# so the factor is EXACTLY ONE. (It is not 31.6 - that would be the factor for
# GPa*sqrt(mm).) Cross-check: Gc = 10 GPa*um, E = 200 GPa
#   -> K = sqrt(Gc*E) = 44.72 GPa*sqrt(um) = 44.72 MPa*sqrt(m),
#   and sqrt(E[Pa] * Gc[J/m^2]) = sqrt(200e9 * 10e3) = 4.472e7 Pa*sqrt(m). Same.
GPA_SQRT_UM_TO_MPA_SQRT_M = 1.0


def K_from_Gc(Gc_GPa_um, E_GPa):
    """K = sqrt(E*Gc) in MPa*sqrt(m) for Gc in GPa*um and E in GPa."""
    return np.sqrt(E_GPa * 1e9 * Gc_GPa_um * 1e3) / 1e6


# --------------------------------------------------------- microstructure ---
class Microstructure:
    """The explicitly modelled EBSD patch and its per-cell stiffness.

    Parameters
    ----------
    npz_path      : micro_<tag>.npz written by 070/preprocess_ebsd_to_grid.py
    cfg           : dict from load_config()
    sfun          : callable s(x_map_um), applied in region SCALED_REGION only
    rotate_ccw90  : rotate the whole patch by +90 deg in the FE frame
                    (transverse crack case). Orientations are rotated with it.
    origin        : (x0, y0) placement of the patch inside the FE domain [um]

    After construction the arrays are in FE ROW ORDER: row 0 = smallest y,
    column 0 = smallest x, both relative to `origin`.
    """

    def __init__(self, npz_path, cfg, sfun=None, rotate_ccw90=False,
                 origin=(0.0, 0.0), verbose=True):
        d = np.load(npz_path)
        euler = d['euler_deg']; phase = d['phase']
        gid = d['grain_id']; zone = d['zone']
        self.meta = json.loads(str(d['meta']))
        self.step = float(self.meta['step_um'])
        self.rotated = bool(rotate_ccw90)
        self.plane = cfg["fracture"]["plane_state"]
        self.origin = np.asarray(origin, dtype=float)

        ny0, nx0 = phase.shape
        # map-frame x of each column (needed for s(x) BEFORE any rotation)
        x_map = np.asarray(d['x_um']) if 'x_um' in d else \
            (np.arange(nx0) + 0.5) * self.step

        extra = X.ROT_Z90 if rotate_ccw90 else None
        C, Ex, s_map, info = self._build(euler, phase, gid, zone, x_map, cfg,
                                         sfun, extra, verbose)

        # map row order -> FE row order (row 0 = bottom), then optional rotation
        arrays = [C, Ex, s_map, zone, phase, gid, euler]
        arrays = [X.to_fe_rows(a) for a in arrays]
        if rotate_ccw90:
            arrays = [X.rotate_grid_ccw90(a) for a in arrays]
        self.C, self.Ex, self.s_map, self.zone, self.phase, self.gid, self.euler = arrays

        self.ny, self.nx = self.phase.shape
        self.Lx = self.nx * self.step
        self.Ly = self.ny * self.step
        self.info = info
        self.info["patch_grid_ny_nx"] = [int(self.ny), int(self.nx)]
        self.info["patch_Lx_um"] = float(self.Lx)
        self.info["patch_Ly_um"] = float(self.Ly)
        self.info["rotated_ccw90"] = self.rotated
        if verbose:
            print(f'Patch {self.ny}x{self.nx} Zellen, {self.Lx:.0f} x {self.Ly:.0f} um, '
                  f'Zellgroesse {self.step:.3f} um, rotiert={self.rotated}')

    # -- per-cell tensors on the (still map-ordered) grid --------------------
    def _build(self, euler, phase, gid, zone, x_map, cfg, sfun, extra, verbose):
        ny, nx = phase.shape
        tab = constants_table(cfg)
        rmat = cfg["region_material"]
        sfun = sfun or (lambda x: 1.0)
        plane = cfg["fracture"]["plane_state"]

        C = np.empty((ny, nx, 3, 3))
        Ex = np.empty((ny, nx))
        s_map = np.ones((ny, nx))
        cache, counts = {}, {}
        for j in range(ny):
            zrow, prow, grow = zone[j], phase[j], gid[j]
            for i in range(nx):
                z = int(zrow[i])
                mat = rmat.get(z, rmat.get(str(z), "316L"))
                cry = PHASE_CRYSTAL.get(int(prow[i]), "fcc")
                key = (int(grow[i]), mat, cry)
                hit = cache.get(key)
                if hit is None:
                    g = X.bunge_to_g(*np.deg2rad(euler[j, i]))
                    C0 = tab[mat][cry] if cry in tab[mat] else next(iter(tab[mat].values()))
                    pre = X.FLIP_X180 if extra is None else extra @ X.FLIP_X180
                    C2 = X.reduce_C(X.rotate_C(C0, g, pre_rot=pre), plane)
                    hit = (C2, X.E_directional(C2))
                    cache[key] = hit
                    counts[(mat, cry)] = counts.get((mat, cry), 0) + 1
                s = float(np.asarray(sfun(x_map[i]))) if z == SCALED_REGION else 1.0
                C[j, i] = s * hit[0]
                Ex[j, i] = s * hit[1]
                s_map[j, i] = s

        info = {"n_distinct_grain_tensors": len(cache),
                "grains_per_material_crystal": {f"{m}/{c}": n
                                                for (m, c), n in sorted(counts.items())},
                "region_material": {str(k): v for k, v in rmat.items()},
                "single_crystal_cubic_GPa": cfg["single_crystal_cubic_GPa"],
                "plane_state": plane,
                "E_x_range_GPa": [float(Ex.min()), float(Ex.max())]}
        if verbose:
            print(f'Materialzuordnung: {info["grains_per_material_crystal"]}')
            print(f'  {len(cache)} verschiedene Korn-Steifigkeitstensoren; '
                  f'E_x pro Zelle {Ex.min():.0f}...{Ex.max():.0f} GPa')
        return C, Ex, s_map, info

    def place(self, x_left=0.0, y_center=0.0):
        """Position the patch in the FE domain: left edge at `x_left`, vertical
        centre at `y_center` (the crack line). Must match the geometry written
        by `mesh_fracture_micro.py`, which puts the patch at
        [0, Lx] x [-Ly/2, +Ly/2]."""
        self.origin = np.array([x_left, y_center - 0.5 * self.Ly])
        return self

    # -- point queries -------------------------------------------------------
    def indices(self, x, y):
        """(j, i, inside) for FE coordinates x, y (arrays)."""
        xr = (np.asarray(x) - self.origin[0]) / self.step
        yr = (np.asarray(y) - self.origin[1]) / self.step
        i = np.floor(xr).astype(np.int64)
        j = np.floor(yr).astype(np.int64)
        inside = (i >= 0) & (i < self.nx) & (j >= 0) & (j < self.ny)
        return np.clip(j, 0, self.ny - 1), np.clip(i, 0, self.nx - 1), inside

    def sample_averaged(self, x, y, h, cfg, scheme='hill'):
        """Element-wise EFFECTIVE stiffness instead of a nearest-pixel draw.

        A point query is only meaningful while the element is smaller than a
        grain. In this microstructure the median grain of the 17-4PH zone has an
        equivalent diameter of 5.4 um = 1.6 EBSD pixels, so for h = 12 um the
        centroid lookup hands almost every element a grain none of its
        neighbours share: the stiffness field becomes element-scale white noise
        between 94 and 300 GPa. That is not resolved microstructure, it wrecks
        the conditioning, and it is what makes Newton stall.

        Here every element instead gets the average over the pixels it actually
        covers (a box of side h around its centroid):

            voigt : <C>            (stiff bound)
            reuss : <C^-1>^-1      (compliant bound)
            hill  : (voigt+reuss)/2

        Grains larger than h are unaffected - the box then lies inside a single
        grain and the average is that grain's tensor. Only the sub-element
        grains get homogenised, which is the honest thing to do with material
        the mesh cannot represent anyway.
        """
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        h = np.atleast_1d(np.asarray(h, dtype=float))
        if h.size == 1:
            h = np.full(x.shape, float(h[0]))
        base = self.sample(x, y, cfg)
        if scheme == 'none':
            base['n_pixels_averaged'] = np.ones(x.size)
            return base

        # half-width in cells, at least 0 (single pixel -> identical to sample)
        half = np.maximum(np.floor(0.5 * h / self.step).astype(int), 0)
        j0, i0, inside = self.indices(x, y)
        C = base['C']
        npix = np.ones(x.size)
        for k in np.flatnonzero(inside & (half > 0)):
            a, b = max(j0[k] - half[k], 0), min(j0[k] + half[k] + 1, self.ny)
            c, e = max(i0[k] - half[k], 0), min(i0[k] + half[k] + 1, self.nx)
            block = self.C[a:b, c:e].reshape(-1, 3, 3)
            npix[k] = len(block)
            if len(block) < 2:
                continue
            voigt = block.mean(axis=0)
            if scheme == 'voigt':
                C[k] = voigt
                continue
            reuss = np.linalg.inv(np.linalg.inv(block).mean(axis=0))
            C[k] = reuss if scheme == 'reuss' else 0.5 * (voigt + reuss)
        base['C'] = C
        base['Ex'] = np.where(inside,
                              [X.E_directional(c) for c in C], base['Ex'])
        base['n_pixels_averaged'] = npix
        return base

    def sample(self, x, y, cfg):
        """Stiffness and bookkeeping fields at FE points (x, y).

        Points outside the patch get the isotropic embedding tensor and the
        marker values region=-1, grain_id=-1, phase=0, s=1.
        """
        j, i, inside = self.indices(x, y)
        n = len(np.atleast_1d(x))
        Cemb = embedding_C2D(cfg)
        C = np.repeat(Cemb[None, :, :], n, axis=0)
        C[inside] = self.C[j[inside], i[inside]]

        Ex = np.full(n, X.E_directional(Cemb))
        Ex[inside] = self.Ex[j[inside], i[inside]]
        zone = np.full(n, -1.0);  zone[inside] = self.zone[j[inside], i[inside]]
        phase = np.zeros(n);      phase[inside] = self.phase[j[inside], i[inside]]
        gidf = np.full(n, -1.0);  gidf[inside] = self.gid[j[inside], i[inside]]
        s = np.ones(n);           s[inside] = self.s_map[j[inside], i[inside]]
        return dict(C=C, Ex=Ex, region=zone, phase=phase, grain_id=gidf,
                    s=s, inside=inside.astype(float))


def make_sfun(expr):
    """s(x) from a numpy expression string in the MAP x coordinate [um]."""
    return eval('lambda x: ' + expr, {'np': np})   # noqa: S307 (user-provided)


if __name__ == '__main__':
    print(describe(load_config()))
