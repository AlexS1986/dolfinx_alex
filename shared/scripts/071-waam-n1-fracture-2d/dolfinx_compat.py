"""Tiny compatibility shim so 071 runs on dolfinx 0.7.3 (the container image,
`dolfinx/dolfinx:v0.7.3`) as well as on 0.8/0.9 (cluster images).

Only the function-space factory really differs for what 071 needs
(FunctionSpace -> functionspace). Element construction follows the pattern
that is already proven inside this repo (067/run_simulation.py):
`basix.ufl.element("P"|"DP", domain.basix_cell(), degree, shape=(...))`.
"""
import dolfinx as dlfx
import basix.ufl as bxu


def functionspace(domain, element):
    """dlfx.fem.functionspace (0.8+) or dlfx.fem.FunctionSpace (0.7)."""
    factory = getattr(dlfx.fem, "functionspace", None) or dlfx.fem.FunctionSpace
    return factory(domain, element)


def scalar_element(domain, degree=1, family="P"):
    return bxu.element(family, domain.basix_cell(), degree, shape=())


def vector_element(domain, degree=1, family="P", dim=None):
    return bxu.element(family, domain.basix_cell(), degree,
                       shape=(dim or domain.geometry.dim,))


def tensor_element(domain, shape, degree=0, family="DP"):
    return bxu.element(family, domain.basix_cell(), degree, shape=shape)


def mixed_element(elements):
    return bxu.mixed_element(elements)
