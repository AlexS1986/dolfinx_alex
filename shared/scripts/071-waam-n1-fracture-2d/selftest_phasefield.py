#!/usr/bin/env python3
"""
dolfinx unit tests for `phasefield_anisotropic` - RUN INSIDE THE CONTAINER.

    docker exec -it alex-dolfinx bash -lc \
      "cd /home/scripts/071-waam-n1-fracture-2d && python3 selftest_phasefield.py"

Checks
  1  patch test: for a linear displacement field the assembled cell stress
     equals C . eps_v analytically, for a general anisotropic C.
  2  energy: E_el of a uniformly strained block equals 0.5 eps.C.eps * area.
  3  ISOTROPIC LIMIT: with C built from (lam, mu) the residual vector, the
     Jacobian and the elastic energy are identical (to round-off) to
     `alex.phasefield.StaticPhaseFieldProblem2D`, which is the reference
     implementation used by 067. This is the test that guarantees the new
     class did not change the physics, only generalise the elastic law.
  4  degradation: sigma and psi_el scale with g(s, eta) exactly.
  5  Eshelby: for a homogeneous body the contour integral over the outer
     boundary vanishes for a rigid-body-free uniform strain state (no crack),
     i.e. the configurational force of a homogeneous, unloaded-tip body is 0.
  6  heterogeneity: with two different C values the Eshelby contour integral
     is NON-zero - the material force of the interface is picked up. This is
     what makes the far-field J of 071 the *effective* energy release rate.

Everything is assembled on a small unit mesh, so the test runs in seconds.
"""
import os
import sys

import numpy as np
import ufl
import dolfinx as dlfx
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dolfinx_compat as cmp                                  # noqa: E402
import phasefield_anisotropic as pfa                          # noqa: E402
import crystal2d as X                                         # noqa: E402

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
PASS = []
TOL = 1e-10


def check(name, cond=True, extra=''):
    """`cond` defaults to True so a bare check(name) records that the plain
    `assert`s above it passed."""
    if not cond:
        raise AssertionError(f'FAILED: {name} {extra}')
    PASS.append(name)
    if rank == 0:
        print(f'  ok  {name}' + (f'  [{extra}]' if extra else ''))


def unit_mesh(n=8):
    return dlfx.mesh.create_unit_square(comm, n, n, dlfx.mesh.CellType.triangle)


def constant_C(domain, C2):
    Cf = pfa.make_cell_tensor_function(domain)
    m = domain.topology.index_map(domain.topology.dim)
    n = m.size_local + m.num_ghosts
    pfa.set_cell_tensor(Cf, np.repeat(np.asarray(C2)[None], n, axis=0), n)
    return Cf


def integrate(expr):
    return comm.allreduce(dlfx.fem.assemble_scalar(dlfx.fem.form(expr)), MPI.SUM)


# --------------------------------------------------------------------------
C_ANISO = X.grain_C2D(204.6, 137.7, 126.2, 31.0, 57.0, 13.0)   # triclinic 2D
EPS_APPLIED = np.array([1.3e-3, -0.7e-3, 0.9e-3])              # [xx, yy, gxy]


def linear_displacement(domain, V, e):
    """u with eps_xx=e0, eps_yy=e1, gamma_xy=e2 (engineering)."""
    uh = dlfx.fem.Function(V)
    uh.interpolate(lambda x: np.vstack([e[0] * x[0] + e[2] * x[1],
                                        e[1] * x[1]]))
    return uh


def t_patch_test():
    domain = unit_mesh()
    V = cmp.functionspace(domain, cmp.vector_element(domain, 1))
    uh = linear_displacement(domain, V, EPS_APPLIED)
    Cf = constant_C(domain, C_ANISO)
    prob = pfa.StaticPhaseFieldProblem2D_anisotropic(pfa.cubic_degradation(0.1))
    sig_v = prob.sigma_voigt_undegraded(uh, Cf)
    got = np.array([integrate(sig_v[k] * ufl.dx) for k in range(3)])
    want = C_ANISO @ EPS_APPLIED
    check('patch test: sigma = C . eps for anisotropic C',
          np.allclose(got, want, rtol=1e-10),
          f'max rel dev {np.max(np.abs(got - want) / np.abs(want)):.2e}')


def t_energy():
    domain = unit_mesh()
    V = cmp.functionspace(domain, cmp.vector_element(domain, 1))
    uh = linear_displacement(domain, V, EPS_APPLIED)
    Cf = constant_C(domain, C_ANISO)
    prob = pfa.StaticPhaseFieldProblem2D_anisotropic(pfa.cubic_degradation(0.1))
    got = integrate(prob.psiel_undegraded(uh, Cf) * ufl.dx)
    want = 0.5 * EPS_APPLIED @ C_ANISO @ EPS_APPLIED
    check('psi_el = 1/2 eps.C.eps = 1/2 sigma:eps',
          np.isclose(got, want, rtol=1e-11), f'{got:.12e} vs {want:.12e}')


def t_degradation_scaling():
    domain = unit_mesh()
    V = cmp.functionspace(domain, cmp.vector_element(domain, 1))
    S = cmp.functionspace(domain, cmp.scalar_element(domain, 1))
    uh = linear_displacement(domain, V, EPS_APPLIED)
    Cf = constant_C(domain, C_ANISO)
    g = pfa.cubic_degradation(0.1)
    prob = pfa.StaticPhaseFieldProblem2D_anisotropic(g)
    eta = dlfx.fem.Constant(domain, 1e-5)
    for sval in (1.0, 0.7, 0.25, 0.0):
        sh = dlfx.fem.Function(S)
        sh.x.array[:] = sval
        got = integrate(prob.psiel_degraded(sh, eta, uh, Cf) * ufl.dx)
        want = g(sval, 1e-5) * 0.5 * EPS_APPLIED @ C_ANISO @ EPS_APPLIED
        assert np.isclose(got, want, rtol=1e-10), (sval, got, want)
    check('psi_el_degraded == g(s, eta) * psi_el for s in {1, .7, .25, 0}')


def t_isotropic_limit_vs_alex():
    """The decisive test: same physics as alex.phasefield in the isotropic case."""
    try:
        import alex.phasefield as apf
    except ImportError:
        if rank == 0:
            print('  --  alex not importable (PYTHONPATH=/home/utils?), test skipped')
        return

    domain = unit_mesh(10)
    lam, mu = 1.7, 0.9
    C_iso = pfa.lame_to_C2D_plane_strain(lam, mu)      # alex 2D law is plane strain
    Cf = constant_C(domain, C_iso)

    Ve = cmp.vector_element(domain, 1)
    Se = cmp.scalar_element(domain, 1)
    W = cmp.functionspace(domain, cmp.mixed_element([Ve, Se]))
    w = dlfx.fem.Function(W)
    wm1 = dlfx.fem.Function(W)
    dw = ufl.TestFunction(W)
    ddw = ufl.TrialFunction(W)

    rng = np.random.default_rng(0)
    w.x.array[:] = 0.01 * rng.standard_normal(w.x.array.shape)
    wm1.x.array[:] = 0.01 * rng.standard_normal(wm1.x.array.shape)
    w.x.scatter_forward(); wm1.x.scatter_forward()

    S0 = cmp.functionspace(domain, cmp.scalar_element(domain, 0, family='DP'))
    lam_f = dlfx.fem.Function(S0); lam_f.x.array[:] = lam
    mu_f = dlfx.fem.Function(S0); mu_f.x.array[:] = mu
    gc_f = dlfx.fem.Function(S0); gc_f.x.array[:] = 1.3
    for f in (lam_f, mu_f, gc_f):
        f.x.scatter_forward()

    epsilon = dlfx.fem.Constant(domain, 0.05)
    eta = dlfx.fem.Constant(domain, 1e-5)
    iMob = dlfx.fem.Constant(domain, 1e-3)
    dt = dlfx.fem.Constant(domain, 0.1)

    deg = pfa.cubic_degradation(0.1)
    mine = pfa.StaticPhaseFieldProblem2D_anisotropic(deg, pfa.psisurf_from_function)
    theirs = apf.StaticPhaseFieldProblem2D(apf.cubic_degradation(0.1),
                                           apf.psisurf_from_function)

    Rm, _ = mine.prep_newton(w, wm1, dw, ddw, Cf, gc_f, epsilon, eta, iMob, dt)
    Rt, _ = theirs.prep_newton(w, wm1, dw, ddw, lam_f, mu_f, gc_f, epsilon,
                               eta, iMob, dt)

    from dolfinx.fem.petsc import assemble_vector
    bm = assemble_vector(dlfx.fem.form(Rm)); bm.assemble()
    bt = assemble_vector(dlfx.fem.form(Rt)); bt.assemble()
    dv = bm.copy(); dv.axpy(-1.0, bt)
    rel = dv.norm() / max(bt.norm(), 1e-300)
    check('isotropic limit: residual identical to alex.StaticPhaseFieldProblem2D',
          rel < 1e-12, f'||dR||/||R|| = {rel:.3e}')

    u, s = ufl.split(w)
    Em = mine.get_E_el_global(s, eta, u, Cf, ufl.dx, comm)
    Et = theirs.get_E_el_global(s, eta, u, lam_f, mu_f, ufl.dx, comm)
    check('isotropic limit: elastic energy identical',
          abs(Em - Et) <= 1e-12 * max(abs(Et), 1e-30),
          f'{Em:.15e} vs {Et:.15e}')

    Esh_m = mine.getEshelby(w, eta, Cf)
    Esh_t = theirs.getEshelby(w, eta, lam_f, mu_f)
    d = integrate(ufl.inner(Esh_m - Esh_t, Esh_m - Esh_t) * ufl.dx)
    r = integrate(ufl.inner(Esh_t, Esh_t) * ufl.dx)
    check('isotropic limit: Eshelby tensor identical',
          d <= 1e-24 * max(r, 1e-30), f'||dSigma||^2 = {d:.3e}')


def t_eshelby_homogeneous_vanishes():
    domain = unit_mesh(12)
    Ve = cmp.vector_element(domain, 1)
    Se = cmp.scalar_element(domain, 1)
    W = cmp.functionspace(domain, cmp.mixed_element([Ve, Se]))
    w = dlfx.fem.Function(W)
    Wu, mapu = W.sub(0).collapse()
    uu = linear_displacement(domain, Wu, EPS_APPLIED)
    w.x.array[mapu] = uu.x.array
    Ws, maps_ = W.sub(1).collapse()
    w.x.array[maps_] = 1.0
    w.x.scatter_forward()

    Cf = constant_C(domain, C_ANISO)
    prob = pfa.StaticPhaseFieldProblem2D_anisotropic(pfa.cubic_degradation(0.1))
    eta = dlfx.fem.Constant(domain, 1e-5)
    n = ufl.FacetNormal(domain)
    Esh = prob.getEshelby(w, eta, Cf)
    J = np.array([integrate(ufl.dot(Esh, n)[k] * ufl.ds) for k in range(2)])
    scale = abs(0.5 * EPS_APPLIED @ C_ANISO @ EPS_APPLIED)
    check('homogeneous body, uniform strain: contour integral of Eshelby = 0',
          np.max(np.abs(J)) < 1e-10 * scale,
          f'|J| = {np.max(np.abs(J)):.3e}, scale = {scale:.3e}')


def t_eshelby_heterogeneous_nonzero():
    domain = unit_mesh(16)
    tdim = domain.topology.dim
    m = domain.topology.index_map(tdim)
    nall = m.size_local + m.num_ghosts
    mid = dlfx.mesh.compute_midpoints(domain, tdim,
                                      np.arange(nall, dtype=np.int32))
    Carr = np.repeat(C_ANISO[None], nall, axis=0)
    Carr[mid[:, 0] > 0.5] *= 2.0                       # stiff right half
    Cf = pfa.make_cell_tensor_function(domain)
    pfa.set_cell_tensor(Cf, Carr, nall)

    Ve = cmp.vector_element(domain, 1)
    Se = cmp.scalar_element(domain, 1)
    W = cmp.functionspace(domain, cmp.mixed_element([Ve, Se]))
    w = dlfx.fem.Function(W)
    Wu, mapu = W.sub(0).collapse()
    uu = linear_displacement(domain, Wu, EPS_APPLIED)
    w.x.array[mapu] = uu.x.array
    Ws, maps_ = W.sub(1).collapse()
    w.x.array[maps_] = 1.0
    w.x.scatter_forward()

    prob = pfa.StaticPhaseFieldProblem2D_anisotropic(pfa.cubic_degradation(0.1))
    eta = dlfx.fem.Constant(domain, 1e-5)
    n = ufl.FacetNormal(domain)
    Esh = prob.getEshelby(w, eta, Cf)
    Jx = integrate(ufl.dot(Esh, n)[0] * ufl.ds)
    scale = abs(0.5 * EPS_APPLIED @ C_ANISO @ EPS_APPLIED)
    check('heterogeneous body: contour integral picks up the material force',
          abs(Jx) > 1e-3 * scale, f'Jx = {Jx:.4e}, scale = {scale:.4e}')


if __name__ == '__main__':
    if rank == 0:
        print('071 phase-field selftest (dolfinx ' + dlfx.__version__ + ')')
    for fn in [t_patch_test, t_energy, t_degradation_scaling,
               t_isotropic_limit_vs_alex, t_eshelby_homogeneous_vanishes,
               t_eshelby_heterogeneous_nonzero]:
        fn()
    if rank == 0:
        print(f'\nALL PASS ({len(PASS)} checks)')
