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


def figure_view_and_section(path, wsteg, name):
    """Fig. 15 of the 6x3x2 series: a) 3D view (box outline, surface mesh of the front face z = z_max showing the refined crack band,
    all pore surfaces), y vertical as in the v2.0 figure; b) section through the pore centres z = -w_c/2 (resolved region)."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    w_c = wsteg + 1.0
    xyz, tets = read_msh41(path)
    f = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
    u, c = np.unique(f, axis=0, return_counts=True)
    bf = u[c == 1]; cen = xyz[bf].mean(axis=1); lo, hi = xyz.min(0), xyz.max(0)
    on_box = np.any((np.abs(cen - lo) < 1e-6) | (np.abs(cen - hi) < 1e-6), axis=1)
    pores, front = bf[~on_box], bf[np.abs(cen[:, 2] - hi[2]) < 1e-6]
    P = lambda tri: xyz[tri][:, :, [0, 2, 1]]          # plot axes (x, z, y): y vertical
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
    fig = plt.figure(figsize=(6.3, 3.9))
    ax = fig.add_axes([0.02, -0.04, 0.46, 1.08], projection="3d")
    fe = P(front); segs = np.concatenate([fe[:, [0, 1]], fe[:, [1, 2]], fe[:, [2, 0]]])
    ax.add_collection3d(Line3DCollection(segs, colors=(0.25, 0.25, 0.6, 0.35), linewidths=0.12, rasterized=True))
    ax.add_collection3d(Poly3DCollection(P(pores), facecolor=(0.3, 0.35, 0.8, 0.85), edgecolor=(0.1, 0.1, 0.35, 0.6),
                                         linewidths=0.08, rasterized=True))
    X, Y, Z = (lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2])
    for a in X:
        for b in Y:
            ax.plot([a, a], [Z[0], Z[1]], [b, b], "k-", lw=0.5)
        for b in Z:
            ax.plot([a, a], [b, b], [Y[0], Y[1]], "k-", lw=0.5)
    for a in Y:
        for b in Z:
            ax.plot([X[0], X[1]], [b, b], [a, a], "k-", lw=0.5)
    ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]), zoom=0.95)
    ax.set_xlim(*X); ax.set_ylim(*Z); ax.set_zlim(*Y); ax.view_init(elev=14, azim=-58); ax.set_axis_off()
    o = np.array([lo[0] - 6.0, hi[2] + 2.0, lo[1] - 1.0]); Ls = 3.5     # coordinate triad, left of the box
    for d, lab in ((np.array([1, 0, 0]), "x"), (np.array([0, 0, 1]), "y"), (np.array([0, -1, 0]), "z")):
        ax.quiver(*o, *(Ls * d), color="k", lw=0.8, arrow_length_ratio=0.25)
        ax.text(*(o + 1.25 * Ls * d), f"${lab}$")
    ax.text2D(0.03, 0.95, "a)", transform=ax.transAxes)
    ax2 = fig.add_axes([0.56, 0.27, 0.43, 0.5])
    polys = plane_section(xyz, tets, -0.5 * w_c); sg = segments(polys, -lo[0])
    m = (np.abs(sg[:, :, 1]).min(axis=1) < 3.3)
    ax2.add_collection(LineCollection(sg[m], colors="k", linewidths=0.15, rasterized=True))
    ax2.axhline(0.0, color="tab:red", lw=0.6, ls="--")
    ax2.set_xlim(0, hi[0] - lo[0]); ax2.set_ylim(-3.3, 3.3); ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x/L$"); ax2.set_ylabel(r"$y/L$")
    ax2.text(0.0, 1.06, "b)", transform=ax2.transAxes)
    fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=400)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    print("wrote", name, f"({len(tets)} tets, {len(pores)} pore faces, {len(front)} front faces)")
