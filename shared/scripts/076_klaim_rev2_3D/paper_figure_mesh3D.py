#!/usr/bin/env python3
"""
Fig. 15 (Section 5.1): section of the 3D fixed-mesh tetrahedral mesh (S1F, w_s = 0.375L, h_band = 0.1L) through the plane of the
pore centres z = -w_c/2, a) whole cross-section, b) resolved porous region. Reads the gmsh 4.1 ASCII file domain_mesh.msh of the
case folder (copy it from $HPC_SCRATCH/076_klaim_rev2_3D/S1F_3D_vM_ws0.375/), no gmsh/dolfinx needed.
Usage: python3 paper_figure_mesh3D.py path/to/domain_mesh.msh [wsteg]
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "046_plasticity", "KLAIM_2025", "revision_2", "pics")


def read_msh41(path):
    """nodes (dict tag -> xyz as array with tag index) and tetrahedra (element type 4) of an ASCII gmsh 4.1 file"""
    with open(path) as f:
        lines = f.read().split("\n")
    i = lines.index("$Nodes") + 1
    n_blocks, n_nodes, _, max_tag = map(int, lines[i].split()); i += 1
    xyz = np.zeros((max_tag + 1, 3))
    for _ in range(n_blocks):
        _, _, parametric, n = map(int, lines[i].split()); i += 1
        tags = np.array(lines[i:i + n], dtype=int); i += n
        coords = np.array([lines[i + k].split()[:3] for k in range(n)], dtype=float); i += n
        xyz[tags] = coords
    i = lines.index("$Elements") + 1
    n_blocks = int(lines[i].split()[0]); i += 1
    tets = []
    for _ in range(n_blocks):
        _, _, etype, n = map(int, lines[i].split()); i += 1
        if etype == 4:
            tets.append(np.array([lines[i + k].split()[1:5] for k in range(n)], dtype=int))
        i += n
    return xyz, np.vstack(tets)


def plane_section(xyz, tets, z0):
    """polygons (3 or 4 points, x-y) of the intersection of all tetrahedra with the plane z = z0"""
    P = xyz[tets]                      # (n, 4, 3)
    d = P[:, :, 2] - z0
    cut = (d.min(axis=1) < 0) & (d.max(axis=1) > 0)
    P, d = P[cut], d[cut]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    polys = []
    for p, dd in zip(P, d):
        pts = []
        for a, b in edges:
            if dd[a] * dd[b] < 0:
                s = dd[a] / (dd[a] - dd[b])
                pts.append(p[a, :2] + s * (p[b, :2] - p[a, :2]))
        pts = np.array(pts)
        c = pts.mean(axis=0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
        polys.append(pts)
    return polys


def segments(polys, x_shift):
    segs = []
    for p in polys:
        q = p + np.array([x_shift, 0.0])
        segs.extend(np.stack([q, np.roll(q, -1, axis=0)], axis=1))
    return np.array(segs)


def main():
    path = sys.argv[1]
    wsteg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.375
    w_c = wsteg + 1.0
    xyz, tets = read_msh41(path)
    z0 = -0.5 * w_c                       # plane of the pore centres (2 cells in z, w_c/4 embedding on each side)
    polys = plane_section(xyz, tets, z0)
    x_shift = 3.0 * w_c                   # x measured from the left end of the domain (length 6 w_c), as in Sections 4 and 5
    segs = segments(polys, x_shift)
    print(f"{len(tets)} tetrahedra, {len(polys)} cut by z = {z0:.4f}")
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.6), gridspec_kw={"width_ratios": [1.0, 2.35]})
    for ax, lw, lim in ((axes[0], 0.15, (0, 6 * w_c, -10, 10)), (axes[1], 0.25, (0.6 * w_c, 5.4 * w_c, -2.6, 2.6))):
        x0, x1, y0, y1 = lim
        m = ((segs[:, :, 0].max(axis=1) >= x0) & (segs[:, :, 0].min(axis=1) <= x1)
             & (segs[:, :, 1].max(axis=1) >= y0) & (segs[:, :, 1].min(axis=1) <= y1))
        ax.add_collection(LineCollection(segs[m], colors="k", linewidths=lw, rasterized=True))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
        ax.set_xlabel(r"$x/L$"); ax.set_ylabel(r"$y/L$")
    axes[1].axhline(0.0, color="tab:red", lw=0.6, ls="--")
    axes[0].add_patch(plt.Rectangle((0.6 * w_c, -2.6), 4.8 * w_c, 5.2, fill=False, ec="tab:blue", lw=0.8))
    for ax, lab in zip(axes, ("a)", "b)")):
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, va="top", ha="left", bbox=dict(fc="w", ec="none", pad=1))
    fig.tight_layout()
    name = f"PAPER_rev2_3D_mesh_section_ws{wsteg:g}"
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=450)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    print("wrote", name)


if __name__ == "__main__":
    main()
