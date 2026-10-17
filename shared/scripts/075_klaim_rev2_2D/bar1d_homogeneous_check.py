#!/usr/bin/env python3
"""Rate-independent, homogeneous (gradient-free) estimate of the 1D-bar fracture stress of the three parameter sets
[vM], [Eq J_c], [Eq sigma*] -- a numpy-only cross-check of the FE bar in 050 (needs no dolfinx, no scipy).

Bar as in 050/run_simulation_plasticity.py: 2D plane strain (eps_zz = 0), u_y = 0 on all nodes (eps_yy = 0), lam = 0,
mu = mu0 = 1, loaded in x -> uniaxial strain. Model as in alex (StaticPhaseFieldProblem_plasticity_noll):
  degradation g(s) = beta (s^3 - s^2) + 3 s^2 - 2 s^3,  psi_surf = J_c (1-s)^2 / (4 eps),
  s-equation (homogeneous, rate-independent): g'(s) [psi_el + psi_pl] = J_c (1-s) / (2 eps),
  J2 plasticity with linear isotropic hardening on the UNDEGRADED stress, psi_pl = sig_y alpha + H alpha^2 / 2 (undegraded),
  sigma = g(s) * 2 mu (eps - eps_p).
Checks (2026-10-17): elastic J_c = 1: sig* = 1.7230 (FE 050: 1.7237); vM beta = 0.2: sig* = 1.2806 (FE 050, 2026-05: 1.2803).
Elastic: sig* is exactly proportional to sqrt(J_c), so J_c = (sig*_vM / sig*_el(J_c=1))^2 gives the equal-strength set.
Usage: python3 bar1d_homogeneous_check.py [--beta 0.2] [--eps 0.1] [--sig_y 1.0] [--hard 0.2222222] [--gc_eqsig 0.5679]
"""
import argparse

import numpy as np


def brentq(f, a, b, tol=1e-12):
    """bisection (no scipy needed); f(a) and f(b) must have opposite signs"""
    fa, fb = f(a), f(b)
    if fa == 0:
        return a
    if fb == 0:
        return b
    assert fa * fb < 0, "no sign change"
    for _ in range(200):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0 or (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

ap = argparse.ArgumentParser()
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--eps", type=float, default=0.1)
ap.add_argument("--mu", type=float, default=1.0)
ap.add_argument("--sig_y", type=float, default=1.0)
ap.add_argument("--hard", type=float, default=0.2222222)
ap.add_argument("--gc_eqsig", type=float, default=0.5679, help="J_c of [Eq sigma*] to check")
ap.add_argument("--free_y", action="store_true", help="sigma_yy = 0 instead of eps_yy = 0")
args = ap.parse_args()


def g(s, b):
    return b * (s**3 - s**2) + 3 * s**2 - 2 * s**3


def dg(s, b):
    return b * (3 * s**2 - 2 * s) + 6 * s - 6 * s**2


def s_from_psi(psi, b, Gc, eps, s_prev):
    f = lambda s: dg(s, b) * psi - Gc * (1 - s) / (2 * eps)
    grid = np.linspace(s_prev, 1e-6, 4000)
    v = f(grid)
    idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
    if len(idx) == 0:
        return 1e-6 if v[0] > 0 else s_prev
    return brentq(f, grid[idx[0] + 1], grid[idx[0]])


def peak(mu, sig_y, H, Gc, eps, b, elastic, free_y, n=3000, e_max=1.6):
    ep = np.zeros(3)
    alpha, s, eyy = 0.0, 1.0, 0.0
    out = []
    for e in np.linspace(0.0, e_max, n):
        def state(eyy_, ep, alpha):
            eps_tot = np.array([e, eyy_, 0.0])
            sig_tr = 2 * mu * (eps_tot - ep)
            dev = sig_tr - sig_tr.mean()
            q = np.sqrt(1.5 * dev @ dev)
            f = q - (sig_y + H * alpha)
            if elastic or f <= 0:
                return sig_tr, ep, alpha
            dgam = f / (3 * mu + H)
            ep_new = ep + 1.5 * dgam * dev / q
            return 2 * mu * (eps_tot - ep_new), ep_new, alpha + dgam

        if free_y:
            eyy = brentq(lambda y: state(y, ep, alpha)[0][1], eyy - 1.0, eyy + 1.0)
        sig, ep, alpha = state(eyy, ep, alpha)
        eps_el = np.array([e, eyy, 0.0]) - ep
        psi = mu * eps_el @ eps_el + (0.0 if elastic else sig_y * alpha + 0.5 * H * alpha**2)
        s = s_from_psi(psi, b, Gc, eps, s)
        out.append((e, g(s, b) * sig[0], s, alpha))
    out = np.array(out)
    i = np.argmax(out[:, 1])
    return out[i]


bc = "sigma_yy = 0" if args.free_y else "eps_yy = 0 (u_y = 0 as in 050)"
print(f"homogeneous rate-independent bar: lam = 0, mu = {args.mu}, eps = {args.eps}, beta = {args.beta}, {bc}; stresses in sig_y units")
e_el, s_el, _, _ = peak(args.mu, args.sig_y, args.hard, 1.0, args.eps, args.beta, True, args.free_y)
e_vm, s_vm, s_s, a_vm = peak(args.mu, args.sig_y, args.hard, 1.0, args.eps, args.beta, False, args.free_y)
e_es, s_es, _, _ = peak(args.mu, args.sig_y, args.hard, args.gc_eqsig, args.eps, args.beta, True, args.free_y)
print(f"[Eq J_c]     J_c = 1.0     : sig* = {s_el:.4f}  eps* = {e_el:.4f}")
print(f"[vM]  sig_y = {args.sig_y}, H = {args.hard}: sig* = {s_vm:.4f}  eps* = {e_vm:.4f}  (s = {s_s:.3f}, alpha = {a_vm:.3f} at the peak)")
print(f"[Eq sigma*]  J_c = {args.gc_eqsig}: sig* = {s_es:.4f}  eps* = {e_es:.4f}  = {s_es / s_vm:.4f} x sig*_vM")
print(f"J_c for exactly equal strength: (sig*_vM / sig*_el)^2 = {(s_vm / s_el) ** 2:.4f}")
