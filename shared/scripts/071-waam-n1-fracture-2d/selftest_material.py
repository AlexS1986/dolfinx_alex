#!/usr/bin/env python3
"""
Numpy unit tests for the 071 material chain (no dolfinx needed).

    python3 selftest_material.py        ->  "ALL PASS" or a failing assertion

Covers: cubic/isotropic constants, Bond rotation, plane-stress condensation,
the extra 90 deg rotation used for the transverse ROI, the grid rotation index
map, the Voigt energy identity that `phasefield_anisotropic` relies on, and -
if project 070 is reachable - bit-for-bit agreement with its crystal module.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crystal2d as X                                     # noqa: E402
import materials_fracture_2d as MF                        # noqa: E402

PASS = []


def check(name, cond=True, extra=''):
    """`cond` defaults to True so a bare check(name) records that the plain
    `assert`s above it passed."""
    assert cond, f'FAILED: {name} {extra}'
    PASS.append(name)
    print(f'  ok  {name}' + (f'  [{extra}]' if extra else ''))


# --------------------------------------------------------------------------
# 1 - elementary tensors
# --------------------------------------------------------------------------
def t_cubic_isotropic_limit():
    C11, C12 = 250.0, 100.0
    C44 = 0.5 * (C11 - C12)                 # Zener A = 1 -> isotropic
    C = X.cubic_C(C11, C12, C44)
    lam, mu = C12, C44
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))
    check('cubic with A=1 equals isotropic',
          np.allclose(C, X.isotropic_C(E, nu)), f'E={E:.2f} nu={nu:.4f}')


def t_bond_rotation_roundtrip():
    C = X.cubic_C(231.4, 134.7, 116.4)
    rng = np.random.default_rng(7)
    for _ in range(5):
        ang = rng.uniform(0, 360, 3)
        g = X.bunge_to_g(*np.deg2rad(ang))
        check_ = X.rotate_C(X.rotate_C(C, g), g.T)  # rotate there and back
        assert np.allclose(check_, C, atol=1e-8), ang
    check('Bond rotation is invertible', True)


def t_rotation_preserves_invariants():
    C = X.cubic_C(231.4, 134.7, 116.4)
    g = X.bunge_to_g(*np.deg2rad([37.0, 61.0, 12.0]))
    Cr = X.rotate_C(C, g)
    check('rotated C stays symmetric', np.allclose(Cr, Cr.T, atol=1e-9))
    check('rotated C stays positive definite',
          np.all(np.linalg.eigvalsh(Cr) > 0))
    # trace of the 3x3 upper block is a rotation invariant for cubic symmetry
    check('bulk modulus invariant under rotation',
          np.isclose(Cr[:3, :3].sum(), C[:3, :3].sum(), rtol=1e-10),
          f'{Cr[:3,:3].sum():.6f} vs {C[:3,:3].sum():.6f}')


def t_single_crystal_E_extremes():
    C11, C12, C44 = 204.6, 137.7, 126.2
    E100, E110, E111 = X.cubic_E_extremes(C11, C12, C44)
    # <100> along x: identity orientation, plane stress in the (100) plane
    C2 = X.grain_C2D(C11, C12, C44, 0.0, 0.0, 0.0, map_frame=False)
    check('E<100> from condensed tensor matches analytic',
          np.isclose(X.E_directional(C2, 0.0), E100, rtol=1e-10),
          f'{X.E_directional(C2,0.0):.4f} vs {E100:.4f}')
    # <110> is 45 deg from <100> inside the (001) plane
    check('E<110> at 45 deg matches analytic',
          np.isclose(X.E_directional(C2, 45.0), E110, rtol=1e-10),
          f'{X.E_directional(C2,45.0):.4f} vs {E110:.4f}')
    # for Zener A > 1 (both steels here) the ordering is E<100> < E<110> < E<111>
    check('cubic anisotropy ordering E<100> < E<110> < E<111>',
          E100 < E110 < E111, f'{E100:.1f} < {E110:.1f} < {E111:.1f}')
    # and the condensed 2D tensor must reproduce the <100>-<110> spread exactly
    check('in-plane E(theta) spans [E<100>, E<110>] for a cube orientation',
          np.isclose(min(X.E_directional(C2, a) for a in np.linspace(0, 90, 91)),
                     E100, rtol=1e-9)
          and np.isclose(max(X.E_directional(C2, a) for a in np.linspace(0, 90, 91)),
                         E110, rtol=1e-9))


def t_plane_stress_isotropic():
    E, nu = 200.0, 0.3
    C2 = X.reduce_C(X.isotropic_C(E, nu), 'stress')
    check('plane-stress condensation of isotropic C',
          np.allclose(C2, X.isotropic_C2D(E, nu, 'stress')))
    # uniaxial stress test: sig = [1,0,0] -> eps_xx = 1/E, eps_yy = -nu/E
    eps = np.linalg.solve(C2, np.array([1.0, 0.0, 0.0]))
    check('plane stress recovers E and nu',
          np.isclose(1.0 / eps[0], E, rtol=1e-12)
          and np.isclose(-eps[1] / eps[0], nu, rtol=1e-12),
          f'E={1/eps[0]:.6f} nu={-eps[1]/eps[0]:.6f}')


def t_plane_strain_isotropic():
    E, nu = 200.0, 0.3
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    C2 = X.reduce_C(X.isotropic_C(E, nu), 'strain')
    ref = np.array([[lam + 2 * mu, lam, 0.0],
                    [lam, lam + 2 * mu, 0.0],
                    [0.0, 0.0, mu]])
    check('plane-strain reduction equals the Lame form (engineering shear)',
          np.allclose(C2, ref))


# --------------------------------------------------------------------------
# 2 - the 90 degree rotation used for the transverse ROI
# --------------------------------------------------------------------------
def t_rot_z90_orientation():
    C11, C12, C44 = 204.6, 137.7, 126.2
    rng = np.random.default_rng(3)
    for _ in range(20):
        a = rng.uniform(0, 360, 3)
        C0 = X.grain_C2D(C11, C12, C44, *a)
        C90 = X.grain_C2D(C11, C12, C44, *a, extra_rot=X.ROT_Z90)
        assert np.isclose(X.E_directional(C90, 0.0), X.E_directional(C0, 90.0),
                          rtol=1e-10), a
        assert np.isclose(X.E_directional(C90, 90.0), X.E_directional(C0, 0.0),
                          rtol=1e-10), a
    check('ROT_Z90: E_x(rotated) == E_y(original) for random orientations', True)


def t_rot_z90_is_a_rotation():
    C11, C12, C44 = 231.4, 134.7, 116.4
    a = [15.0, 40.0, 70.0]
    C0 = X.grain_C2D(C11, C12, C44, *a)
    C360 = X.grain_C2D(C11, C12, C44, *a,
                       extra_rot=X.rot_z(90) @ X.rot_z(90) @ X.rot_z(90) @ X.rot_z(90))
    check('four 90 deg rotations return the original tensor',
          np.allclose(C0, C360, atol=1e-9))


def t_grid_rotation_index_map():
    ny, nx, h = 5, 8, 2.0
    # store the FE coordinates of each cell centre as the "material"
    j, i = np.mgrid[0:ny, 0:nx]
    xc = (i + 0.5) * h
    yc = (j + 0.5) * h
    A = np.stack([xc, yc], axis=-1)                 # (ny, nx, 2)
    R = X.rotate_grid_ccw90(A)
    check('rotated patch has swapped shape', R.shape == (nx, ny, 2))
    Ly = ny * h
    for jj in range(nx):
        for ii in range(ny):
            x_old, y_old = R[jj, ii]
            # the cell centre it now occupies
            x_new, y_new = (ii + 0.5) * h, (jj + 0.5) * h
            assert np.isclose(x_new, Ly - y_old) and np.isclose(y_new, x_old), \
                (jj, ii, R[jj, ii])
    check('rotate_grid_ccw90 realises (x,y) -> (Ly-y, x)', True)


# --------------------------------------------------------------------------
# 3 - Microstructure end to end on a synthetic npz
# --------------------------------------------------------------------------
def _synthetic_npz(path, ny=6, nx=9, step=3.371, seed=11):
    rng = np.random.default_rng(seed)
    gid = np.arange(ny * nx, dtype=np.int32).reshape(ny, nx)
    euler = rng.uniform(0, 360, (ny, nx, 3))
    phase = rng.integers(1, 3, (ny, nx)).astype(np.int8)
    x_um = (np.arange(nx) + 0.5) * step
    y_um = (np.arange(ny) + 0.5) * step
    zone = np.ones((ny, nx), np.int8)
    meta = ('{"step_um": %r, "frame": "map", "zone": "all transition"}' % step)
    np.savez_compressed(path, euler_deg=euler, phase=phase, grain_id=gid,
                        zone=zone, x_um=x_um, y_um=y_um, meta=meta)
    return dict(ny=ny, nx=nx, step=step, euler=euler, phase=phase, gid=gid)


def t_microstructure_lookup_and_rotation():
    cfg = MF.load_config()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'micro_test.npz')
        ref = _synthetic_npz(p)
        m0 = MF.Microstructure(p, cfg, verbose=False).place()
        m9 = MF.Microstructure(p, cfg, rotate_ccw90=True, verbose=False).place()

        check('unrotated patch size',
              (m0.ny, m0.nx) == (ref['ny'], ref['nx']))
        check('rotated patch size is swapped',
              (m9.ny, m9.nx) == (ref['nx'], ref['ny']))

        h = ref['step']
        # a point query must return the tensor of the grain sitting there
        rng = np.random.default_rng(5)
        for _ in range(30):
            i = int(rng.integers(0, m0.nx)); j = int(rng.integers(0, m0.ny))
            x = (i + 0.5) * h
            y = (j + 0.5) * h - 0.5 * m0.Ly
            got = m0.sample(np.array([x]), np.array([y]), cfg)
            assert got['inside'][0] == 1.0
            assert np.allclose(got['C'][0], m0.C[j, i])
        check('point query hits the right cell (unrotated)', True)

        # a query outside the patch must return the isotropic embedding
        Cemb = MF.embedding_C2D(cfg)
        out = m0.sample(np.array([-5.0 * h]), np.array([0.0]), cfg)
        check('outside the patch -> isotropic embedding',
              out['inside'][0] == 0.0 and np.allclose(out['C'][0], Cemb)
              and out['grain_id'][0] == -1.0)

        # rotated field must equal the rotated tensors of the same grains
        gid_fe = X.to_fe_rows(ref['gid'])
        gid_rot = X.rotate_grid_ccw90(gid_fe)
        check('grain ids follow the grid rotation',
              np.array_equal(m9.gid, gid_rot))
        # E_x of the rotated patch == E_y of the unrotated patch, cell by cell
        Ey0 = np.array([[X.E_directional(m0.C[j, i], 90.0) for i in range(m0.nx)]
                        for j in range(m0.ny)])
        Ex9 = np.array([[X.E_directional(m9.C[j, i], 0.0) for i in range(m9.nx)]
                        for j in range(m9.ny)])
        check('E_x(rotated grid) == E_y(original grid) cell by cell',
              np.allclose(Ex9, X.rotate_grid_ccw90(Ey0), rtol=1e-10))


def t_sfun_prefactor():
    cfg = MF.load_config()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'micro_test.npz')
        _synthetic_npz(p)
        m1 = MF.Microstructure(p, cfg, verbose=False)
        m2 = MF.Microstructure(p, cfg, sfun=MF.make_sfun('2.0'), verbose=False)
        check('s(x)=2 doubles every tensor in the transition zone',
              np.allclose(m2.C, 2.0 * m1.C) and np.allclose(m2.Ex, 2.0 * m1.Ex))
        mg = MF.Microstructure(p, cfg, sfun=MF.make_sfun('1+0.5*np.exp(-((x-10.)/5.)**2)'),
                               verbose=False)
        check('gauss s(x) stays within [1, 1.5]',
              mg.s_map.min() >= 1.0 - 1e-12 and mg.s_map.max() <= 1.5 + 1e-12,
              f'{mg.s_map.min():.4f}..{mg.s_map.max():.4f}')


# --------------------------------------------------------------------------
# 4 - the Voigt identity the FE model relies on
# --------------------------------------------------------------------------
def t_voigt_energy_identity():
    """0.5 * eps_v . C . eps_v  ==  0.5 * sigma : eps  with ENGINEERING shear."""
    rng = np.random.default_rng(2)
    C2 = X.grain_C2D(204.6, 137.7, 126.2, 23.0, 45.0, 67.0)
    for _ in range(20):
        e_xx, e_yy, g_xy = rng.normal(size=3) * 1e-3
        ev = np.array([e_xx, e_yy, g_xy])
        sv = C2 @ ev
        sig = np.array([[sv[0], sv[2]], [sv[2], sv[1]]])
        eps = np.array([[e_xx, 0.5 * g_xy], [0.5 * g_xy, e_yy]])
        assert np.isclose(0.5 * ev @ sv, 0.5 * np.sum(sig * eps), rtol=1e-12)
    check('Voigt energy == 0.5 sigma:eps (engineering shear bookkeeping)', True)


def t_energy_positive():
    C2 = X.grain_C2D(231.4, 134.7, 116.4, 12.0, 88.0, 200.0)
    check('condensed 2D tensor is positive definite',
          np.all(np.linalg.eigvalsh(0.5 * (C2 + C2.T)) > 0))


# --------------------------------------------------------------------------
# 5 - cross-check against project 070 (if reachable)
# --------------------------------------------------------------------------
def t_cross_check_070():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(os.path.dirname(here), '070-waam-n1-transition-2d')
    if not os.path.isfile(os.path.join(cand, 'plane_stress_crystal.py')):
        print('  --  070 not reachable, cross-check skipped')
        return
    sys.path.insert(0, cand)
    import plane_stress_crystal as P70                      # noqa
    rng = np.random.default_rng(19)
    for _ in range(25):
        a = rng.uniform(0, 360, 3)
        c = rng.uniform(100, 260, 3)
        mine = X.grain_C2D(c[0], c[1], c[2], *a)
        theirs = P70.grain_C2D(c[0], c[1], c[2], *a)
        assert np.allclose(mine, theirs, rtol=0, atol=1e-11), a
    check('crystal2d reproduces 070/plane_stress_crystal exactly', True)


if __name__ == '__main__':
    print('071 material selftest')
    for fn in [t_cubic_isotropic_limit, t_bond_rotation_roundtrip,
               t_rotation_preserves_invariants, t_single_crystal_E_extremes,
               t_plane_stress_isotropic, t_plane_strain_isotropic,
               t_rot_z90_orientation, t_rot_z90_is_a_rotation,
               t_grid_rotation_index_map, t_microstructure_lookup_and_rotation,
               t_sfun_prefactor, t_voigt_energy_identity, t_energy_positive,
               t_cross_check_070]:
        fn()
    print(f'\nALL PASS ({len(PASS)} checks)')
