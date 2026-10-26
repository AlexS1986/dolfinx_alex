#!/usr/bin/env python3
"""Check and render a 3D porous mesh (gmsh 4.1 ASCII): pore count, pore-surface resolution per row, 3D view of all pore surfaces
inside the box outline (as in the v2.0 figure) and the section through the pore centres.
Usage: python3 mesh_check_render.py domain_mesh.msh wsteg out_prefix"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import LineCollection
from paper_figure_mesh3D import read_msh41, plane_section, segments

path, wsteg, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
w_c = wsteg + 1.0
xyz, tets = read_msh41(path)
f = np.sort(np.vstack([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]]), axis=1)
u, c = np.unique(f, axis=0, return_counts=True)
bf = u[c == 1]; cen = xyz[bf].mean(axis=1); lo, hi = xyz.min(0), xyz.max(0)
pore = ~np.any((np.abs(cen - lo) < 1e-6) | (np.abs(cen - hi) < 1e-6), axis=1)
pf, pc = bf[pore], cen[pore]
# pore index by nearest cell centre
nx = int(round((hi[0] - lo[0]) / w_c)) - 2
ix = np.round((pc[:, 0] - (lo[0] + w_c + 0.5 * w_c)) / w_c).astype(int)
iy = np.round(pc[:, 1] / w_c).astype(int); iz = np.round((pc[:, 2] + 0.5 * w_c) / w_c).astype(int)
keys = np.stack([ix, iy, iz], 1); pores = np.unique(keys, axis=0)
edge = np.linalg.norm(xyz[pf[:, 0]] - xyz[pf[:, 1]], axis=1)
print(f"{len(tets)} tets, domain {np.round(lo,3)} .. {np.round(hi,3)}, {len(pores)} pores ({nx} x {len(np.unique(iy))} x {len(np.unique(iz))})")
for r in sorted(np.unique(iy)):
    m = iy == r
    print(f"  pore row y = {r * w_c:+.3f}: {m.sum() // max(1, len(np.unique(keys[m], axis=0)))} faces per pore, mean edge {edge[m].mean():.3f} L")
fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(111, projection="3d")
tri = xyz[pf]
ax.add_collection3d(Poly3DCollection(tri, facecolor=(0.35, 0.4, 0.85, 0.55), edgecolor=(0.1, 0.1, 0.4, 0.5), linewidths=0.15))
X = [lo[0], hi[0]]; Y = [lo[1], hi[1]]; Z = [lo[2], hi[2]]
for a in X:
    for b in Y:
        ax.plot([a, a], [b, b], Z, "k-", lw=0.6)
    for b in Z:
        ax.plot([a, a], Y, [b, b], "k-", lw=0.6)
for a in Y:
    for b in Z:
        ax.plot(X, [a, a], [b, b], "k-", lw=0.6)
ax.set_xlim(lo[0], hi[0]); ax.set_ylim(-0.5 * (hi[0] - lo[0]), 0.5 * (hi[0] - lo[0])); ax.set_zlim(lo[2] - 2, hi[2] + 2)
ax.set_box_aspect((hi[0] - lo[0], hi[0] - lo[0], hi[2] - lo[2] + 4)); ax.view_init(elev=18, azim=-60)
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.set_title(f"pore surfaces, w_s = {wsteg:g}L, {len(pores)} pores, {len(tets)} tets", fontsize=9)
fig.tight_layout(); fig.savefig(out + "_3D.png", dpi=170); plt.close(fig)
polys = plane_section(xyz, tets, -0.5 * w_c); segs = segments(polys, -lo[0])
fig, ax = plt.subplots(figsize=(9, 3.6))
m = (np.abs(segs[:, :, 1]).min(axis=1) < 3.2)
ax.add_collection(LineCollection(segs[m], colors="k", linewidths=0.2)); ax.set_xlim(0, hi[0] - lo[0]); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
ax.set_title(f"section z = -w_c/2 through the pore centres, w_s = {wsteg:g}L", fontsize=9); ax.set_xlabel("x/L"); ax.set_ylabel("y/L")
fig.tight_layout(); fig.savefig(out + "_section.png", dpi=170); plt.close(fig)
print("wrote", out + "_3D.png", out + "_section.png")
