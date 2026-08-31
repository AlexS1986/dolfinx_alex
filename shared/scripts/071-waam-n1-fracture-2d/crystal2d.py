"""
Numpy-only crystal-elasticity helpers for the 2D phase-field FRACTURE model
(071-waam-n1-fracture-2d). No dolfinx imports -> unit-testable anywhere.

Provenance
----------
The rotation / condensation core is the same mathematics as
`070-waam-n1-transition-2d/plane_stress_crystal.py`. It is COPIED here (not
imported) so that 071 is self-contained and 070 stays untouched; the identity
of both implementations is checked in `selftest_material.py` whenever the 070
folder is reachable. Do not "improve" one without the other.

Added on top of 070:
  * `rot_z(theta_deg)`            - rotation of the sample frame about z
  * `ROT_Z90`                     - the 90 deg case used for the transverse ROI
  * `rotate_grid_ccw90`           - rotate a microstructure array patch
  * `isotropic_C2D`               - plane-stress / plane-strain isotropic 3x3
  * `voigt3_to_tensor` helpers    - bookkeeping shared with the FE code

Conventions (identical to 069/070 and alex.linearelastic)
  * 3D Voigt order [xx, yy, zz, yz, xz, xy], ENGINEERING shear strains.
  * 2D Voigt order [xx, yy, xy], engineering shear.
  * Bunge Euler angles (phi1, Phi, phi2), v_crystal = g @ v_sample.
  * TSL/OIM map frame has y DOWN; the FE frame has y UP. Conversion via a
    180 deg rotation about x (FLIP_X180). Never "fix" this by negating angles.
"""
import numpy as np

VOIGT_A = np.array([0, 1, 5])   # in-plane block (xx, yy, xy)
VOIGT_B = np.array([2, 3, 4])   # condensed block (zz, yz, xz)

# 180-degree rotation about x: maps TSL map frame (y down, z into plane)
# onto the FE frame (y up, z out of plane). Right-handed both ways.
FLIP_X180 = np.diag([1.0, -1.0, -1.0])


def rot_z(theta_deg):
    """Rotation of the SAMPLE frame about z by theta (deg), counter-clockwise.

    A material direction with coordinates v in the old frame has coordinates
    rot_z(theta) @ v after the body has been rotated by +theta about z.
    """
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])


ROT_Z90 = rot_z(90.0)


def cubic_C(C11, C12, C44):
    """Cubic single-crystal stiffness, 6x6 Voigt (order xx,yy,zz,yz,xz,xy)."""
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def isotropic_C(E, nu):
    """Isotropic 3D stiffness 6x6: C11=lam+2mu, C12=lam, C44=mu."""
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return cubic_C(lam + 2 * mu, lam, mu)


def bunge_to_g(phi1, Phi, phi2):
    """Bunge Euler (rad) -> g with v_crystal = g @ v_sample."""
    c1, s1, c, s, c2, s2 = (np.cos(phi1), np.sin(phi1), np.cos(Phi),
                            np.sin(Phi), np.cos(phi2), np.sin(phi2))
    return np.array([
        [c1 * c2 - s1 * s2 * c,  s1 * c2 + c1 * s2 * c, s2 * s],
        [-c1 * s2 - s1 * c2 * c, -s1 * s2 + c1 * c2 * c, c2 * s],
        [s1 * s,                 -c1 * s,                c]])


def bond_matrix(a):
    """6x6 stress Bond matrix M with C_sample = M @ C_crystal @ M.T,
    a = rotation crystal->sample (a = g^T). Voigt order xx,yy,zz,yz,xz,xy."""
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


def rotate_C(C6, g, pre_rot=None):
    """C in the sample frame for crystal orientation g (crystal<-sample).

    pre_rot maps MAP-frame coordinates onto the FE-frame coordinates of the
    same material direction:  v_fe = pre_rot @ v_map.  Since
    v_crystal = g @ v_map = g @ pre_rot^T @ v_fe  we get g_eff = g @ pre_rot^T.
    """
    if pre_rot is not None:
        g = g @ pre_rot.T
    M = bond_matrix(g.T)
    return M @ C6 @ M.T


def plane_stress_condense(C6):
    """3D 6x6 Voigt stiffness -> 2D 3x3 plane-stress stiffness [xx,yy,xy].

    Enforces sigma_zz = sigma_yz = sigma_xz = 0 by static condensation:
      C_red = C_AA - C_AB @ inv(C_BB) @ C_BA
    Valid for arbitrary (triclinic) C6; out-of-plane strains relax freely."""
    A, B = VOIGT_A, VOIGT_B
    CAA = C6[np.ix_(A, A)]
    CAB = C6[np.ix_(A, B)]
    CBB = C6[np.ix_(B, B)]
    return CAA - CAB @ np.linalg.solve(CBB, CAB.T)


def plane_strain_reduce(C6):
    """3D 6x6 Voigt stiffness -> 2D 3x3 plane-STRAIN stiffness [xx,yy,xy].

    eps_zz = eps_yz = eps_xz = 0, so the in-plane block is simply extracted.
    Provided for completeness / comparison; 071 runs plane stress by default
    (consistent with 070)."""
    A = VOIGT_A
    return C6[np.ix_(A, A)].copy()


def reduce_C(C6, plane='stress'):
    """Dispatch to plane_stress_condense / plane_strain_reduce."""
    if plane == 'stress':
        return plane_stress_condense(C6)
    if plane == 'strain':
        return plane_strain_reduce(C6)
    raise ValueError(f'unknown plane state {plane!r} (use "stress"/"strain")')


def grain_C2D(C11, C12, C44, phi1_deg, Phi_deg, phi2_deg,
              map_frame=True, extra_rot=None, plane='stress'):
    """Full chain for one grain: cubic constants + Bunge angles (deg, TSL map
    frame) -> 3x3 stiffness in the FE frame (y up).

    extra_rot : optional additional sample-frame rotation applied AFTER the
                map->FE flip (e.g. ROT_Z90 for the rotated transverse ROI).
    """
    g = bunge_to_g(*np.deg2rad([phi1_deg, Phi_deg, phi2_deg]))
    pre = FLIP_X180 if map_frame else np.eye(3)
    if extra_rot is not None:
        pre = extra_rot @ pre
    C6 = rotate_C(cubic_C(C11, C12, C44), g, pre_rot=pre)
    return reduce_C(C6, plane)


def isotropic_C2D(E, nu, plane='stress'):
    """Isotropic 3x3 stiffness [xx,yy,xy] with engineering shear.

    plane stress:  C = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]
    plane strain:  in-plane block of the 3D isotropic tensor.
    """
    if plane == 'stress':
        f = E / (1.0 - nu ** 2)
        return np.array([[f, f * nu, 0.0],
                         [f * nu, f, 0.0],
                         [0.0, 0.0, f * (1.0 - nu) / 2.0]])
    return reduce_C(isotropic_C(E, nu), 'strain')


# --------------------------------------------------------------------------
# Engineering constants of a 2D tensor (for checks / reporting)
# --------------------------------------------------------------------------
def E_directional(C2, theta_deg=0.0):
    """Young's modulus of a 3x3 plane-stress stiffness in direction theta
    (deg from x). E = 1 / S'_11 with S' the rotated compliance."""
    S = np.linalg.inv(C2)
    c, s = np.cos(np.deg2rad(theta_deg)), np.sin(np.deg2rad(theta_deg))
    s11 = (S[0, 0] * c ** 4 + S[1, 1] * s ** 4
           + (2 * S[0, 1] + S[2, 2]) * c ** 2 * s ** 2
           + 2 * (S[0, 2] * c ** 2 + S[1, 2] * s ** 2) * c * s)
    return 1.0 / s11


def cubic_E_extremes(C11, C12, C44):
    """Analytic single-crystal E<100>, E<110>, E<111> (GPa) for cubic."""
    S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
    S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
    S44 = 1.0 / C44
    J = S11 - S12 - 0.5 * S44
    return 1 / S11, 1 / (S11 - J * 0.5), 1 / (S11 - 2 * J / 3)


# --------------------------------------------------------------------------
# Grid rotation (transverse ROI)
# --------------------------------------------------------------------------
def to_fe_rows(arr):
    """MAP-style array (row 0 = TOP of the map, y down) -> FE-style array
    (row 0 = smallest y_fe).  Just a row flip; works for (ny,nx,...)."""
    return arr[::-1]


def rotate_grid_ccw90(arr_fe):
    """Rotate an FE-style array patch by +90 deg about z (counter-clockwise).

    A cell sitting at FE position (x, y) ends up at (-y, x); after shifting
    back into the positive quadrant the index map is

        new[j' = i, i' = ny-1-j] = old[j, i]      i.e.   new = old.T[:, ::-1]

    for the leading two axes, so trailing axes (e.g. the 3 Euler angles) are
    carried along unchanged. The matching orientation transform is ROT_Z90
    applied AFTER FLIP_X180 (see `grain_C2D(extra_rot=...)`).
    """
    a = np.asarray(arr_fe)
    ax = list(range(a.ndim))
    ax[0], ax[1] = ax[1], ax[0]
    return np.transpose(a, ax)[:, ::-1]


def from_fe_rows(arr_fe):
    """Inverse of `to_fe_rows` (FE-style -> MAP-style)."""
    return arr_fe[::-1]
