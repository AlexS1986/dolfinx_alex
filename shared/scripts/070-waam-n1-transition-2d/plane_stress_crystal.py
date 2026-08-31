"""
Numpy-only crystal-elasticity helpers for the 2D plane-stress transition-zone
model (070-waam-n1-transition-2d). No dolfinx imports -> unit-testable anywhere.

Conventions (match 069/waam_crystal.py and alex.linearelastic):
  * 3D Voigt order [xx, yy, zz, yz, xz, xy], ENGINEERING shear strains.
  * 2D (plane stress, x-y plane) Voigt order [xx, yy, xy], engineering shear.
  * Bunge Euler angles (phi1, Phi, phi2), v_crystal = g @ v_sample.
  * Sample frame of the FE model: x = horizontal map axis (= load axis),
    y = UPWARD vertical map axis, z = out of plane.  TSL/OIM scan frames have
    y pointing DOWN (image rows); the reader converts orientations with a
    180-degree rotation about x (FLIP_X180) so everything downstream is in the
    FE frame.
"""
import numpy as np

VOIGT_A = np.array([0, 1, 5])   # in-plane block (xx, yy, xy)
VOIGT_B = np.array([2, 3, 4])   # condensed block (zz, yz, xz)

# 180-degree rotation about x: maps TSL map frame (y down, z into plane)
# onto the FE frame (y up, z out of plane). Right-handed both ways.
FLIP_X180 = np.diag([1.0, -1.0, -1.0])


def cubic_C(C11, C12, C44):
    """Cubic single-crystal stiffness, 6x6 Voigt (order xx,yy,zz,yz,xz,xy)."""
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    C[0, 0] = C[1, 1] = C[2, 2] = C11
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def isotropic_C(E, nu):
    """Isotropic stiffness 6x6 (for tests): C11=lam+2mu, C12=lam, C44=mu."""
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
    pre_rot: optional extra rotation applied to the SAMPLE frame (e.g.
    FLIP_X180 to go from the TSL map frame to the FE frame):
    v_crystal = g @ v_map = g @ pre_rot^T @ v_fe -> g_eff = g @ pre_rot^T."""
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


def grain_C2D(C11, C12, C44, phi1_deg, Phi_deg, phi2_deg, map_frame=True):
    """Full chain for one grain: cubic constants + Bunge angles (deg, TSL map
    frame) -> 3x3 plane-stress stiffness in the FE frame (y up)."""
    g = bunge_to_g(*np.deg2rad([phi1_deg, Phi_deg, phi2_deg]))
    C6 = rotate_C(cubic_C(C11, C12, C44), g,
                  pre_rot=FLIP_X180 if map_frame else None)
    return plane_stress_condense(C6)


# --------------------------------------------------------------------------
# Reference engineering constants (for checks / reporting)
# --------------------------------------------------------------------------
def cubic_E_extremes(C11, C12, C44):
    """Analytic single-crystal E<100>, E<110>, E<111> (GPa) for cubic."""
    S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
    S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
    S44 = 1.0 / C44
    J = S11 - S12 - 0.5 * S44
    return 1 / S11, 1 / (S11 - J * 0.5), 1 / (S11 - 2 * J / 3)
