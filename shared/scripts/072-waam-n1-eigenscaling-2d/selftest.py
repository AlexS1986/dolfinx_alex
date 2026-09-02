"""Unit checks for plane_stress_crystal.py (numpy only). Run: python3 selftest.py"""
import numpy as np
from plane_stress_crystal import (cubic_C, isotropic_C, bunge_to_g, bond_matrix,
                                  rotate_C, plane_stress_condense, grain_C2D,
                                  cubic_E_extremes, FLIP_X180)

rng = np.random.default_rng(0)
ok = True
def check(name, cond):
    global ok
    print(('PASS' if cond else 'FAIL'), name)
    ok = ok and cond

# 1. isotropic C is rotation-invariant
Ci = isotropic_C(200.0, 0.3)
g = bunge_to_g(*rng.uniform(0, 2*np.pi, 3))
check('isotropic invariant under rotation',
      np.allclose(rotate_C(Ci, g), Ci, atol=1e-9))

# 2. plane-stress condensation of isotropic C == textbook plane-stress matrix
E, nu = 200.0, 0.3
Cps = plane_stress_condense(Ci)
ref = E/(1-nu**2)*np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
check('isotropic plane stress matches E/(1-nu^2) formula',
      np.allclose(Cps, ref, atol=1e-9))

# 3. cubic invariance under 90-degree rotations
Cc = cubic_C(204.6, 137.7, 126.2)
g90 = bunge_to_g(np.pi/2, 0, 0)
check('cubic invariant under 90deg rotation', np.allclose(rotate_C(Cc, g90), Cc, atol=1e-8))

# 4. uniaxial along rotated <100>/<111>: E from condensed 1x1 further condensation
def E_axial_from_C6(C6):
    # sigma = C eps with only eps_xx prescribed, all other stresses zero:
    S = np.linalg.inv(C6)
    return 1.0 / S[0, 0]
E100, E110, E111 = cubic_E_extremes(204.6, 137.7, 126.2)
check('E<100> via compliance', np.isclose(E_axial_from_C6(Cc), E100, rtol=1e-9))
# rotate so that <111> || x: g maps sample->crystal; choose g with first ROW? g@ex = crystal dir of sample x
# build rotation a (crystal->sample) with columns = crystal axes in sample frame... simpler:
# find g such that g @ [1,0,0] = [1,1,1]/sqrt3: use rotation mapping ex->n in crystal space
n = np.array([1., 1, 1])/np.sqrt(3)
v = np.cross([1., 0, 0], n); s = np.linalg.norm(v); c = n[0]
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
Rg = np.eye(3) + K + K@K*((1-c)/s**2)   # rotates ex onto n; use as g
check('E<111> via rotated C', np.isclose(E_axial_from_C6(rotate_C(Cc, Rg)), E111, rtol=1e-6))

# 5. random-orientation average of plane-stress C tends to in-plane isotropy
Cs = np.zeros((3, 3))
N = 4000
for _ in range(N):
    phi1, phi2 = rng.uniform(0, 2*np.pi, 2)
    Phi = np.arccos(rng.uniform(-1, 1))
    Cs += grain_C2D(204.6, 137.7, 126.2, *np.rad2deg([phi1, Phi, phi2]))
Cs /= N
iso_ok = (abs(Cs[0, 0]-Cs[1, 1]) < 1.5 and abs(Cs[0, 2]) < 1.0 and abs(Cs[1, 2]) < 1.0
          and abs(Cs[2, 2] - 0.5*(Cs[0, 0]-Cs[0, 1])) < 2.5)
check('random-texture average ~ in-plane isotropic (Voigt sense)', iso_ok)

# 6. condensation consistency: full 6x6 solve with sigma_B=0 == condensed solve
C6r = rotate_C(Cc, bunge_to_g(0.3, 0.7, 1.1))
epsA = np.array([1e-3, -2e-4, 3e-4])
Cred = plane_stress_condense(C6r)
sigA = Cred @ epsA
# full: unknown eps_B from sigma_B = 0
from plane_stress_crystal import VOIGT_A as A, VOIGT_B as B
CAB = C6r[np.ix_(A, B)]; CBB = C6r[np.ix_(B, B)]; CAA = C6r[np.ix_(A, A)]
epsB = -np.linalg.solve(CBB, CAB.T @ epsA)
sig_full = CAA @ epsA + CAB @ epsB
check('condensed == full solve with free out-of-plane strains',
      np.allclose(sigA, sig_full, atol=1e-12))

# 7. FLIP_X180 is a proper rotation and self-consistent double application
check('FLIP_X180 proper rotation', np.isclose(np.linalg.det(FLIP_X180), 1.0))
C_flip2 = rotate_C(rotate_C(Cc, np.eye(3), pre_rot=FLIP_X180), np.eye(3), pre_rot=FLIP_X180)
check('flip twice = identity on C', np.allclose(C_flip2, Cc, atol=1e-12))

# 8. symmetry of condensed matrix
check('condensed C symmetric', np.allclose(Cred, Cred.T, atol=1e-12))
# positive definite
check('condensed C positive definite', np.all(np.linalg.eigvalsh(Cred) > 0))


# ---------------------------------------------------------------------------
# 9-12. Voigt-Rotation: die Bond-Matrix gegen die exakte Tensorrotation
#       4. Stufe pruefen. Hintergrund: In Voigt-Notation transformieren
#       Spannung und Verzerrung mit ZWEI VERSCHIEDENEN Matrizen,
#       sigma' = M_sig sigma und eps' = M_eps eps, und es gilt
#       C' = M_sig C M_eps^{-1}. Weil fuer Drehungen die Dualitaet
#       M_eps^{-1} = M_sig^T gilt, faellt das mit der hier benutzten Form
#       C' = M_sig C M_sig^T zusammen. M_sig selbst ist NICHT orthogonal;
#       erst in Mandel-Notation (sqrt(2)-Skalierung) wird die
#       Transformationsmatrix orthogonal. Diese Tests belegen beides.
# ---------------------------------------------------------------------------
VOIGT_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def _C6_to_C4(C6):
    C4 = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(VOIGT_PAIRS):
        for J, (k, l) in enumerate(VOIGT_PAIRS):
            for (a_, b_) in {(i, j), (j, i)}:
                for (c_, d_) in {(k, l), (l, k)}:
                    C4[a_, b_, c_, d_] = C6[I, J]
    return C4


def _C4_to_C6(C4):
    return np.array([[C4[i, j, k, l] for (k, l) in VOIGT_PAIRS]
                     for (i, j) in VOIGT_PAIRS])


def _M_sigma(a):
    """Spannungs-Transformationsmatrix, direkt aus der Tensorrotation."""
    M = np.zeros((6, 6))
    for J, (k, l) in enumerate(VOIGT_PAIRS):
        T = np.zeros((3, 3)); T[k, l] = T[l, k] = 1.0
        Tr = a @ T @ a.T
        M[:, J] = [Tr[i, j] for (i, j) in VOIGT_PAIRS]
    return M


def _M_eps(a):
    """Verzerrungs-Transformationsmatrix mit TECHNISCHEN Gleitungen."""
    M = np.zeros((6, 6))
    for J, (k, l) in enumerate(VOIGT_PAIRS):
        T = np.zeros((3, 3))
        if k == l:
            T[k, k] = 1.0
        else:
            T[k, l] = T[l, k] = 0.5          # gamma = 1  ->  eps = 1/2
        Tr = a @ T @ a.T
        M[:, J] = [Tr[i, j] if i == j else 2 * Tr[i, j] for (i, j) in VOIGT_PAIRS]
    return M


Cc4 = cubic_C(204.6, 137.7, 126.2)
worst = 0.0
for _ in range(50):
    gg = bunge_to_g(*rng.uniform(0, 2 * np.pi, 3))
    aa = gg.T
    worst = max(worst, np.abs(bond_matrix(aa) @ Cc4 @ bond_matrix(aa).T
                              - _C4_to_C6(np.einsum('ip,jq,kr,ls,pqrs->ijkl',
                                                    aa, aa, aa, aa, _C6_to_C4(Cc4)))).max())
check('Bond-Rotation == exakte Tensorrotation 4. Stufe (50 Orientierungen)', worst < 1e-9)

aa = bunge_to_g(0.7, 1.1, 2.3).T
Ms, Me = _M_sigma(aa), _M_eps(aa)
check('bond_matrix ist die Spannungs-Transformationsmatrix M_sigma',
      np.allclose(Ms, bond_matrix(aa), atol=1e-12))
check('M_sigma ist NICHT orthogonal (Voigt, keine Mandel-Skalierung)',
      not np.allclose(Ms.T @ Ms, np.eye(6), atol=1e-6))
check('Dualitaet M_sigma^T @ M_eps = I  ->  M_eps^{-1} = M_sigma^T',
      np.allclose(Ms.T @ Me, np.eye(6), atol=1e-12))

# Energie-Dualitaet explizit: die Arbeitsdichte ist in Voigt-Notation das
# Skalarprodukt der beiden Vektoren und bleibt unter Drehung erhalten. Genau
# daraus folgt M_sigma^T M_eps = I (siehe Bericht, Abschnitt 2.4).
Ssym = rng.normal(size=(3, 3)); Ssym = Ssym + Ssym.T
Esym = rng.normal(size=(3, 3)); Esym = Esym + Esym.T
sV = np.array([Ssym[i, j] for (i, j) in VOIGT_PAIRS])
eV = np.array([Esym[i, j] if i == j else 2 * Esym[i, j] for (i, j) in VOIGT_PAIRS])
check('Arbeitsdichte: sigma:eps == sigma_V . eps_V (technische Gleitungen)',
      np.isclose(np.einsum('ij,ij->', Ssym, Esym), sV @ eV))
check('Arbeitsdichte ist drehinvariant: (M_sig s).(M_eps e) == s.e',
      np.isclose((Ms @ sV) @ (Me @ eV), sV @ eV))

D = np.diag([1, 1, 1, np.sqrt(2), np.sqrt(2), np.sqrt(2)])       # Voigt -> Mandel
Mm = D @ Ms @ np.linalg.inv(D)
check('in Mandel-Notation ist die Transformationsmatrix orthogonal',
      np.allclose(Mm.T @ Mm, np.eye(6), atol=1e-12))
check('Mandel-Weg liefert dieselbe rotierte Steifigkeit wie der Voigt-Bond-Weg',
      np.allclose(np.linalg.inv(D) @ (Mm @ (D @ Cc4 @ D) @ Mm.T) @ np.linalg.inv(D),
                  Ms @ Cc4 @ Ms.T, atol=1e-9))


# ===========================================================================
# 13+. Projekt 072: Eigenzerlegung des kubischen Tensors (K, C', C44) und
#      separate Skalierung der drei irreduziblen Anteile.
# ===========================================================================
from plane_stress_crystal import (cubic_to_KCpC44, cubic_from_KCpC44,
                                  cubic_eigen_parts, cubic_scaled)

C11, C12, C44v = 204.6, 137.7, 126.2
K, Cp, C44m = cubic_to_KCpC44(C11, C12, C44v)

# 13. roundtrip constants <-> irreducible moduli
check('K/C\'/C44 roundtrip', np.allclose(cubic_from_KCpC44(K, Cp, C44m),
                                         (C11, C12, C44v), atol=1e-12))

# 14. the three parts sum to the full cubic tensor
Ch, Ct, Cs2 = cubic_eigen_parts(C11, C12, C44v)
check('eigen parts sum to cubic C', np.allclose(Ch + Ct + Cs2,
                                                cubic_C(C11, C12, C44v), atol=1e-12))

# 15. 6x6 eigenvalues scale as (aK*3K, aCp*2C' [not checked as matrix eig],
#     ...): as VOIGT-MATRIX eigenvalues: 3K (x1), C11-C12=2C' (x2), C44 (x3)
aK, aCp, aC44 = 1.7, 0.8, 1.3
Csc = cubic_scaled(C11, C12, C44v, aK, aCp, aC44)
ev = np.sort(np.linalg.eigvalsh(Csc))
ev_ref = np.sort([3*K*aK, 2*Cp*aCp, 2*Cp*aCp,
                  C44v*aC44, C44v*aC44, C44v*aC44])
check('scaled Voigt-matrix eigenvalues = (3K aK, 2C\' aCp x2, C44 aC44 x3)',
      np.allclose(ev, ev_ref, atol=1e-9))

# 16. scaled tensor equals combining the scaled parts (linearity)
check('cubic_scaled == aK*Ch + aCp*Ct + aC44*Cs',
      np.allclose(Csc, aK*Ch + aCp*Ct + aC44*Cs2, atol=1e-12))

# 17. equal factors reproduce the 070 scalar prefactor EXACTLY through the
#     full chain (rotation + plane-stress condensation are homogeneous in C)
s = 1.33
gtest = bunge_to_g(0.4, 0.9, 1.7)
C_a = plane_stress_condense(rotate_C(cubic_scaled(C11, C12, C44v, s, s, s),
                                     gtest, pre_rot=FLIP_X180))
C_b = s * plane_stress_condense(rotate_C(cubic_C(C11, C12, C44v),
                                         gtest, pre_rot=FLIP_X180))
check('aK=aCp=aC44=s == s * C (070 equivalence, full chain)',
      np.allclose(C_a, C_b, atol=1e-9))

# 18. Zener ratio scales as aC44/aCp
C11s, C12s, C44s = cubic_from_KCpC44(aK*K, aCp*Cp, aC44*C44m)
A0 = C44v / Cp
A1 = C44s / ((C11s - C12s)/2.0)
check('Zener A scales by aC44/aCp', np.isclose(A1, A0 * aC44/aCp, rtol=1e-12))

# 19. positive factors keep the rotated+condensed tensor positive definite
pd_ok = True
for _ in range(20):
    fa = rng.uniform(0.2, 3.0, 3)
    gg = bunge_to_g(*rng.uniform(0, 2*np.pi, 3))
    Cr = plane_stress_condense(rotate_C(cubic_scaled(C11, C12, C44v, *fa), gg))
    pd_ok = pd_ok and np.all(np.linalg.eigvalsh(Cr) > 0)
check('positive factors -> positive definite condensed C (20 random)', pd_ok)

# 20. materials_eigen_2d.build_cell_tensors: equal factors == scalar s path
#     on a small synthetic grid; factors 1 outside the scaled region
import materials_eigen_2d as ME
nyt, nxt = 4, 6
rng2 = np.random.default_rng(7)
euler_t = np.rad2deg(np.stack([rng2.uniform(0, 2*np.pi, (nyt, nxt)),
                               np.arccos(rng2.uniform(-1, 1, (nyt, nxt))),
                               rng2.uniform(0, 2*np.pi, (nyt, nxt))], axis=-1))
phase_t = rng2.integers(1, 3, (nyt, nxt)).astype(np.int8)
gid_t = np.arange(nyt*nxt, dtype=np.int32).reshape(nyt, nxt)
zone_t = np.zeros((nyt, nxt), dtype=np.int8)
zone_t[:, 2:4] = 1
zone_t[:, 4:] = 2
x_t = (np.arange(nxt) + 0.5) * 10.0
cfg_t = ME.load_config(path='/nonexistent')          # defaults only
af_eq = ME.make_factor_funs(sfun='1.0 + 0.02*x')
C_eq, Ex_eq, am_eq, _ = ME.build_cell_tensors(euler_t, phase_t, gid_t, zone_t,
                                              x_t, cfg_t, af_eq, verbose=False)
af_1 = ME.make_factor_funs()
C_1, Ex_1, am_1, _ = ME.build_cell_tensors(euler_t, phase_t, gid_t, zone_t,
                                           x_t, cfg_t, af_1, verbose=False)
sx = 1.0 + 0.02 * x_t
ok20 = True
for j in range(nyt):
    for i in range(nxt):
        f = sx[i] if zone_t[j, i] == ME.SCALED_REGION else 1.0
        ok20 = ok20 and np.allclose(C_eq[j, i], f*C_1[j, i], atol=1e-9)
        ok20 = ok20 and np.isclose(Ex_eq[j, i], f*Ex_1[j, i], rtol=1e-9)
check('build_cell_tensors: aK=aCp=aC44=s(x) == s(x)*C, nur in region 1', ok20)
check('factor maps are 1 outside the scaled region',
      all(np.allclose(am_eq[n][zone_t != ME.SCALED_REGION], 1.0)
          for n in ME.FACTOR_NAMES))

# 21. a pure aC44 change moves E<111> much more than E<100> (cubic physics:
#     E<100> = f(S11) depends on C', K only; E<111> is C44-dominated)
E100_a, _, E111_a = cubic_E_extremes(*cubic_from_KCpC44(K, Cp, 1.5*C44m))
E100_0, _, E111_0 = cubic_E_extremes(C11, C12, C44v)
check('aC44 leaves E<100> unchanged, moves E<111>',
      np.isclose(E100_a, E100_0, rtol=1e-12) and E111_a > 1.2*E111_0)

print('\nALL PASS' if ok else '\nSOME CHECKS FAILED')
raise SystemExit(0 if ok else 1)
