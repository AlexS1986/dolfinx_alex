#!/usr/bin/env python3
"""
VERIFICATION TOOL ONLY - its output is NOT a project result.

Standalone numpy/scipy Q1-FEM reference solver for the same 2D plane-stress
problem as solve_plane_stress.py (dolfinx). Its only purpose is to
cross-validate the dolfinx implementation (same mesh, BCs, material law -
`materials_2d.py` is shared). All reported results come from dolfinx.

To keep the two apart, everything written here lands in `verification/`, and
`make_figures.py` never reads that folder.

Regular grid of square Q1 elements, one element per microstructure cell,
per-element 3x3 plane-stress C (Voigt [xx,yy,xy], engineering shear).
BCs: u_x=0 @ x=0, u_x=eps0*Lx @ x=Lx, u_y=0 at node (0,0); top/bottom free.

Writes verification/E_<tag>_ref.json (same keys as the dolfinx script) and,
with --save-fields, verification/fields_<tag>_ref.npz.

Usage: python3 reference_solver_numpy.py --micro micro_<tag>.npz --tag <tag>
"""
import argparse, json, os

import numpy as np
import scipy.sparse as sp

import materials_2d as M2


def q1_B_matrices(h):
    """B (3x8) at the 4 Gauss points of a square Q1 element of size h."""
    gp = np.array([-1, 1]) / np.sqrt(3)
    Bs = []
    for eta in gp:
        for xi in gp:
            dN = np.array([
                [-(1-eta), (1-eta), (1+eta), -(1+eta)],
                [-(1-xi), -(1+xi), (1+xi),  (1-xi)]]) / 4.0 * (2.0/h)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN[0]; B[1, 1::2] = dN[1]
            B[2, 0::2] = dN[1]; B[2, 1::2] = dN[0]
            Bs.append(B)
    return np.array(Bs), (h/2.0)**2   # weight*detJ per gp = 1*(h/2)^2


def solve(micro, cfg, sfun, eps0=1e-3, rtol=1e-10):
    d = np.load(micro)
    euler, phase = d['euler_deg'], d['phase']
    gid, zone = d['grain_id'], d['zone']
    meta = json.loads(str(d['meta']))
    ny, nx = phase.shape
    h = float(meta['step_um'])
    Lx, Ly = nx*h, ny*h
    # FE frame y-up: element (j,i) row j=0 = TOP -> FE row r = ny-1-j
    Cc, Exc, s_map, minfo = M2.build_cell_tensors(
        euler, phase, gid, zone, (np.arange(nx)+0.5)*h, cfg, sfun)
    Cfe = Cc[::-1]                                  # FE row-major bottom-up
    zfe = zone[::-1]; pfe = phase[::-1]
    Exfe = Exc[::-1]; gfe = gid[::-1]; sfe = s_map[::-1]

    nnx, nny = nx+1, ny+1
    nid = lambda r, c: r*nnx + c
    # element connectivity (counter-clockwise: n00,n10,n11,n01)
    r, c = np.mgrid[0:ny, 0:nx]
    n00 = nid(r, c); n10 = nid(r, c+1); n11 = nid(r+1, c+1); n01 = nid(r+1, c)
    conn = np.stack([n00, n10, n11, n01], axis=-1).reshape(-1, 4)
    edof = np.empty((len(conn), 8), dtype=np.int64)
    edof[:, 0::2] = 2*conn; edof[:, 1::2] = 2*conn+1

    Bs, wdet = q1_B_matrices(h)
    CE = Cfe.reshape(-1, 3, 3)
    Ke = np.einsum('gki,ekl,glj->eij', Bs, CE, Bs, optimize=True) * wdet
    rows = np.repeat(edof, 8, axis=1).ravel()
    cols = np.tile(edof, (1, 8)).ravel()
    K = sp.coo_matrix((Ke.ravel(), (rows, cols)),
                      shape=(2*nnx*nny, 2*nnx*nny)).tocsr()

    # Dirichlet
    ndof = 2*nnx*nny
    fix = {}
    for rr in range(nny):
        fix[2*nid(rr, 0)] = 0.0
        fix[2*nid(rr, nnx-1)] = eps0*Lx
    fix[2*nid(0, 0)+1] = 0.0
    fixed = np.array(sorted(fix)); vals = np.array([fix[k] for k in fixed])
    free = np.setdiff1d(np.arange(ndof), fixed)
    u = np.zeros(ndof); u[fixed] = vals
    b = -K[:, fixed] @ vals
    Kff = K[free][:, free]
    try:
        import pyamg
        ml = pyamg.smoothed_aggregation_solver(Kff.tocsr())
        x = ml.solve(b[free], tol=rtol, accel='cg', maxiter=600)
        res = np.linalg.norm(Kff@x - b[free])
        if res > 1e-6*np.linalg.norm(b[free]):
            raise RuntimeError('amg not converged')
    except Exception:
        x = sp.linalg.spsolve(Kff.tocsc(), b[free])
    u[free] = x

    # per-element strain/stress at centroid (avg of gauss points)
    ue = u[edof]
    Bc = Bs.mean(axis=0)
    epsE = np.einsum('ij,ej->ei', Bc, ue)
    sigE = np.einsum('eij,ej->ei', CE, epsE)
    out = {'applied_eps_xx': eps0, 'grid': [ny, nx], 'Lx_um': Lx, 'Ly_um': Ly,
           'materials': minfo,
           'E_apparent_GPa': float(sigE[:, 0].mean()/eps0),
           'nu_xy_apparent': float(-epsE[:, 1].mean()/eps0)}
    zflat = zfe.reshape(-1)
    for z, nm in {0: '17-4PH', 1: 'transition', 2: '316L'}.items():
        m = zflat == z
        if not m.any():
            continue
        out[f'zone_{nm}'] = {
            'area_frac': float(m.mean()),
            'avg_sigma_xx_GPa': float(sigE[m, 0].mean()),
            'avg_eps_xx': float(epsE[m, 0].mean()),
            'E_local_GPa': float(sigE[m, 0].mean()/epsE[m, 0].mean())}
    # local E(x) profile: bins along x, E = sum(sig_xx)/sum(eps_xx) per bin
    xmid = (np.tile(np.arange(nx), ny) + 0.5) * h
    nbin = max(int(round(Lx / 100.0)), 10)          # ~100 um bins
    edges = np.linspace(0.0, Lx, nbin + 1)
    ib = np.clip(np.digitize(xmid, edges) - 1, 0, nbin - 1)
    ssum = np.bincount(ib, weights=sigE[:, 0], minlength=nbin)
    esum = np.bincount(ib, weights=epsE[:, 0], minlength=nbin)
    out['E_profile'] = {'x_um': (0.5 * (edges[:-1] + edges[1:])).tolist(),
                        'E_GPa': (ssum / esum).tolist(),
                        'bin_um': float(Lx / nbin)}
    fields = {'u': u.reshape(nny, nnx, 2), 'sig': sigE.reshape(ny, nx, 3),
              'eps': epsE.reshape(ny, nx, 3), 'zone_fe': zfe, 'phase_fe': pfe,
              'Ex_fe': Exfe, 'grain_fe': gfe, 's_fe': sfe}
    return out, fields


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--micro', required=True)
    ap.add_argument('--config', default=None)
    ap.add_argument('--tag', default='transition2d')
    ap.add_argument('--strain', type=float, default=1e-3)
    ap.add_argument('--sfun', default='1.0')
    ap.add_argument('--save-fields', action='store_true')
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    cfg = M2.load_config(args.config, here)
    print(M2.describe(cfg))
    sfun = eval('lambda x: ' + args.sfun, {'np': np})
    out, fields = solve(args.micro, cfg, sfun, eps0=args.strain)
    out['tag'] = args.tag; out['sfun'] = args.sfun
    out['solver'] = 'numpy-Q1-reference (VERIFICATION ONLY, not a result)'
    print(json.dumps(out, indent=2))
    vdir = os.path.join(here, 'verification')
    os.makedirs(vdir, exist_ok=True)
    json.dump(out, open(os.path.join(vdir, f'E_{args.tag}_ref.json'), 'w'), indent=2)
    print(f'wrote verification/E_{args.tag}_ref.json (nur Verifikation)')
    if args.save_fields:
        np.savez_compressed(os.path.join(vdir, f'fields_{args.tag}_ref.npz'), **fields)
        print(f'wrote verification/fields_{args.tag}_ref.npz')


if __name__ == '__main__':
    main()
