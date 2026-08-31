#!/usr/bin/env python3
"""
Phase-field fracture with a CELL-WISE GENERAL 2D STIFFNESS TENSOR.

This is the one piece 071 adds to the existing machinery. All phase-field
classes in `alex.phasefield` parametrise the elastic law by the two Lame
constants (lam, mu) and are therefore isotropic per cell. Here the elastic law
is a full plane 3x3 stiffness matrix per cell (Voigt [xx, yy, xy], ENGINEERING
shear), so every grain can carry its own rotated single-crystal tensor:

    eps_v(u) = [u_x,x , u_y,y , u_x,y + u_y,x]          (engineering shear)
    sig_v    = g(s, eta) * C(x) . eps_v(u)
    psi_el   = 1/2 * g(s, eta) * eps_v . C(x) . eps_v   ( = 1/2 sig : eps )

C(x) is a DG0 tensor Function, i.e. one constant 3x3 matrix per cell -
exactly the object project 070 already builds for the tensile load case
(`materials_2d.build_cell_tensors` there, `materials_fracture_2d` here).

Deliberate modelling choices (state them in any report):
  * NO tension/compression split. The full elastic energy is degraded, as in
    `alex.phasefield.StaticPhaseFieldProblem2D` and in 067. A split is not
    well defined for a general anisotropic C without extra assumptions.
  * Gc enters through `psisurf(s, Gc, epsilon)` and may be a Function; in this
    study it is deliberately constant everywhere.
  * Irreversibility is enforced the same way as in 067, by the Dirichlet
    condition `alex.phasefield.irreversibility_bc` plus the viscous rate term.

The class is intentionally self-contained (imports only dolfinx/ufl/numpy) so
that `selftest_phasefield.py` can check it against the isotropic reference
implementation in `alex.phasefield` inside the container.

Units: GPa, um. Then Gc in GPa*um is Gc in kJ/m^2.
"""
import numpy as np
import dolfinx as dlfx
import ufl
from mpi4py import MPI

import dolfinx_compat as cmp


# ---------------------------------------------------------------------------
# Degradation and surface-energy functions (same forms as alex.phasefield,
# repeated here so this module has no hard dependency on `alex`).
# ---------------------------------------------------------------------------
def quadratic_degradation():
    def degrad(s, eta):
        return s ** 2 + eta

    def degds(s):
        return 2.0 * s

    degrad.derivative = degds
    degrad.name = "quadratic"
    return degrad


def cubic_degradation(beta=0.1):
    def degrad(s, eta):
        return beta * ((s ** 2) * s - (s ** 2)) + 3.0 * (s ** 2) - 2.0 * (s ** 2) * s + eta

    def degds(s):
        return beta * (3.0 * (s ** 2) - 2.0 * s) + 6.0 * s - 6.0 * s ** 2

    degrad.derivative = degds
    degrad.name = "cubic"
    return degrad


def get_degradation(name, beta=0.1):
    return cubic_degradation(beta) if name == "cubic" else quadratic_degradation()


def psisurf_from_function(s, Gc, epsilon):
    """Gc may be a Function or a Constant; epsilon is a dlfx Constant."""
    return Gc * (((1 - s) ** 2) / (4 * epsilon.value)
                 + epsilon.value * (ufl.dot(ufl.grad(s), ufl.grad(s))))


def surf(s, epsilon):
    return (((1 - s) ** 2) / (4 * epsilon.value)
            + epsilon.value * (ufl.dot(ufl.grad(s), ufl.grad(s))))


def get_surf_area(s, epsilon, dx, comm):
    A = dlfx.fem.assemble_scalar(dlfx.fem.form(surf(s, epsilon) * dx))
    return comm.allreduce(A, MPI.SUM)


# ---------------------------------------------------------------------------
# Voigt bookkeeping
# ---------------------------------------------------------------------------
def eps_voigt(u):
    """2D strain in Voigt form [xx, yy, xy] with ENGINEERING shear."""
    return ufl.as_vector([u[0].dx(0), u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


def voigt_to_tensor(v):
    """[xx, yy, xy] -> symmetric 2x2 tensor."""
    return ufl.as_matrix([[v[0], v[2]],
                          [v[2], v[1]]])


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class StaticPhaseFieldProblem2D_anisotropic:
    """Static (rate-regularised) phase-field fracture, anisotropic per cell.

    Parameters
    ----------
    degradationFunction : callable g(s, eta) with attribute `.derivative`
    psisurf             : callable psisurf(s, Gc, epsilon)
    dx                  : integration measure used for the potential
                          (default ufl.dx)

    The signature of `prep_newton` mirrors `alex.phasefield.
    StaticPhaseFieldProblem2D.prep_newton`, with `lam, mu` replaced by the
    single argument `C` (a DG0 3x3 tensor Function). That keeps the driver
    script interchangeable with the existing isotropic problems.
    """

    def __init__(self, degradationFunction, psisurf=psisurf_from_function,
                 dx=ufl.dx):
        self.degradation_function = degradationFunction
        self.psisurf = psisurf
        self.dx = dx

    # -- elastic law --------------------------------------------------------
    def sigma_voigt_undegraded(self, u, C):
        return ufl.dot(C, eps_voigt(u))

    def sigma_undegraded(self, u, C):
        """Undegraded Cauchy stress as a 2x2 tensor."""
        return voigt_to_tensor(self.sigma_voigt_undegraded(u, C))

    def sigma_degraded(self, u, s, C, eta):
        """sigma = g(s, eta) * C : eps  as a 2x2 tensor."""
        return self.degradation_function(s=s, eta=eta) * self.sigma_undegraded(u, C)

    def psiel_undegraded(self, u, C):
        """1/2 eps_v . C . eps_v  ==  1/2 sigma : eps (engineering shear)."""
        e = eps_voigt(u)
        return 0.5 * ufl.dot(e, ufl.dot(C, e))

    def psiel_degraded(self, s, eta, u, C):
        return self.degradation_function(s, eta) * self.psiel_undegraded(u, C)

    # -- Newton system ------------------------------------------------------
    def prep_newton(self, w, wm1, dw, ddw, C, Gc, epsilon, eta, iMob, delta_t):
        def residuum(u, s, du, ds, sm1):
            pot = (self.psiel_degraded(s, eta, u, C)
                   + self.psisurf(s, Gc, epsilon)) * self.dx
            equi = ufl.derivative(pot, u, du)
            sdrive = ufl.derivative(pot, s, ds)
            rate = (s - sm1) / delta_t * ds * self.dx
            Res = iMob * rate + sdrive + equi
            dResdw = ufl.derivative(Res, w, ddw)
            return [Res, dResdw]

        u, s = ufl.split(w)
        um1, sm1 = ufl.split(wm1)
        du, ds = ufl.split(dw)
        return residuum(u, s, du, ds, sm1)

    # -- configurational mechanics -----------------------------------------
    def getEshelby(self, w, eta, C):
        """Elastic Eshelby (energy-momentum) tensor of the degraded solid.

        Sigma = psi_el_degraded * I - grad(u)^T . sigma_degraded

        NOTE for the heterogeneous case: div(Sigma) != 0 wherever C varies in
        space, so the contour integral over the OUTER boundary is the far-field
        J, i.e. crack-tip driving force PLUS the configurational forces of the
        microstructure inside the contour. That far-field J is exactly the
        quantity whose steady-state plateau defines the effective toughness
        (same convention as 067).
        """
        u, s = ufl.split(w)
        eshelby = (self.psiel_degraded(s, eta, u, C) * ufl.Identity(2)
                   - ufl.dot(ufl.grad(u).T, self.sigma_degraded(u, s, C, eta)))
        return ufl.as_tensor(eshelby)

    # -- global energies ----------------------------------------------------
    def get_E_el_global(self, s, eta, u, C, dx, comm):
        Pi = dlfx.fem.assemble_scalar(
            dlfx.fem.form(self.psiel_degraded(s, eta, u, C) * dx))
        return comm.allreduce(Pi, MPI.SUM)

    def get_E_surf_global(self, s, Gc, epsilon, dx, comm):
        Pi = dlfx.fem.assemble_scalar(
            dlfx.fem.form(self.psisurf(s, Gc, epsilon) * dx))
        return comm.allreduce(Pi, MPI.SUM)

    def get_E_total_global(self, s, eta, u, C, Gc, epsilon, dx, comm):
        return (self.get_E_el_global(s, eta, u, C, dx, comm)
                + self.get_E_surf_global(s, Gc, epsilon, dx, comm))


# ---------------------------------------------------------------------------
# Helpers to get numpy per-cell tensors into a DG0 Function and back out
# ---------------------------------------------------------------------------
def make_cell_tensor_function(domain, name="C2D"):
    """DG0 Function of 3x3 matrices (one constant stiffness per cell)."""
    Q = cmp.functionspace(domain, cmp.tensor_element(domain, shape=(3, 3),
                                                     degree=0, family="DP"))
    f = dlfx.fem.Function(Q, name=name)
    return f


def set_cell_tensor(Cfun, C_per_cell, ncells=None):
    """Write an (ncells, 3, 3) numpy array into a DG0 3x3 Function."""
    flat = Cfun.x.array.reshape(-1, 9)
    n = ncells if ncells is not None else len(C_per_cell)
    flat[:n] = np.asarray(C_per_cell).reshape(-1, 9)[:n]
    Cfun.x.scatter_forward()
    return Cfun


def make_cell_scalar_function(domain, name):
    S0 = cmp.functionspace(domain, cmp.scalar_element(domain, degree=0,
                                                      family="DP"))
    return dlfx.fem.Function(S0, name=name)


def set_cell_scalar(f, values, ncells=None):
    n = ncells if ncells is not None else len(values)
    f.x.array[:n] = np.asarray(values)[:n]
    f.x.scatter_forward()
    return f


# ---------------------------------------------------------------------------
# Convenience: build an isotropic per-cell C (for the embedding and for tests)
# ---------------------------------------------------------------------------
def isotropic_C2D_numpy(E, nu, plane="stress"):
    """3x3 isotropic stiffness [xx, yy, xy], engineering shear."""
    if plane == "stress":
        f = E / (1.0 - nu ** 2)
        return np.array([[f, f * nu, 0.0],
                         [f * nu, f, 0.0],
                         [0.0, 0.0, f * (1.0 - nu) / 2.0]])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])


def lame_to_C2D_plane_strain(lam, mu):
    """The 3x3 matrix that reproduces `alex.linearelastic.sigma_as_tensor`
    in 2D, i.e. sigma = lam tr(eps) I + 2 mu eps (plane strain)."""
    return np.array([[lam + 2 * mu, lam, 0.0],
                     [lam, lam + 2 * mu, 0.0],
                     [0.0, 0.0, mu]])
