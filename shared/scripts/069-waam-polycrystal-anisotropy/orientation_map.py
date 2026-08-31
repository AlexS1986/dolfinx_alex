#!/usr/bin/env python3
"""
Orientation (IPF) map of a mid-plane section of the 316L RVE: each point of a
thin section ⊥ build is coloured by the crystal direction lying along the build
axis (inverse-pole-figure colouring, cubic). Makes the crystal-lattice
orientation visible on the multi-grain structure (report §3.2).

No SciPy needed: the section is rasterised from the in-slab mesh element
centroids, each coloured by its grain's IPF(build) colour.

Output: report/fig_orientation_map.png
Usage:  python3 orientation_map.py [--neper-dir <dir>] [--n 500]
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEPER = os.path.join(HERE, "..", "..", "..", "..", "Meshing", "Neper",
                             "data", "04_anisotropy_waam", "neper_pipeline")


def euler_to_g(phi1, Phi, phi2):
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([[c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s],
                     [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
                     [s1*s,             -c1*s,            c]])


def ipf_color(g, sample_dir=(0, 0, 1.0)):
    d = np.abs(np.asarray(g) @ np.array(sample_dir))
    u, v, w = np.sort(d)
    c = np.array([w - v, v - u, u])       # 001->R, 011->G, 111->B
    return c / max(c.max(), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neper-dir", default=DEFAULT_NEPER)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    nd = args.neper_dir

    eul = np.loadtxt(os.path.join(nd, "grain_ori_316L.txt"), comments="#", usecols=(0, 1, 2, 3))
    gcol = {int(r[0]): ipf_color(euler_to_g(*np.deg2rad(r[1:4]))) for r in eul}

    lines = open(os.path.join(nd, f"waam_316L_n{args.n}.msh")).read().split("\n")
    i = lines.index("$Nodes"); nn = int(lines[i + 1]); P = np.zeros((nn + 1, 3))
    for k in range(i + 2, i + 2 + nn):
        p = lines[k].split(); P[int(p[0])] = [float(p[1]), float(p[2]), float(p[3])]
    j = lines.index("$Elements"); ne = int(lines[j + 1])
    G = []; C = []
    for k in range(j + 2, j + 2 + ne):
        p = lines[k].split()
        if len(p) < 3 or p[1] != '4': continue
        nt = int(p[2]); g = int(p[3]); ndd = [int(x) for x in p[3 + nt:3 + nt + 4]]
        G.append(g); C.append(P[ndd].mean(0))
    G = np.array(G); C = np.array(C)
    z0 = (C[:, 2].min() + C[:, 2].max()) / 2
    d = (C[:, 2].max() - C[:, 2].min()) * 0.04
    m = np.abs(C[:, 2] - z0) < d
    xs = C[m, 0]; ys = C[m, 1]; cols = np.array([gcol[g] for g in G[m]])

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(12, 4.4),
                                  gridspec_kw={"width_ratios": [3.4, 1]})
    ax.scatter(xs, ys, c=cols, s=14, marker="s", edgecolors="none")
    ax.set_aspect("equal"); ax.set_xlabel("Schweißrichtung x [µm]")
    ax.set_ylabel("Wandnormale y [µm]")
    ax.set_title("Orientierungskarte (IPF ∥ Aufbau) — Schnitt ⊥ Aufbaurichtung, 316L-RVE",
                 fontsize=10)

    # IPF colour-key triangle (001 red, 101 green, 111 blue)
    a = np.linspace(0, 1, 70)
    A = np.array([0, 0.]); B = np.array([1, 0.]); Cc = np.array([1, 1.]) / np.sqrt(2)
    for u in a:
        for v in a:
            if u + v <= 1:
                pos = A + u * (B - A) + v * (Cc - A)
                dirv = ((1 - u - v) * np.array([0, 0, 1.]) +
                        u * np.array([1, 0, 1.]) / np.sqrt(2) +
                        v * np.array([1, 1, 1.]) / np.sqrt(3))
                dirv /= np.linalg.norm(dirv)
                axl.plot(*pos, "s", ms=4, color=ipf_color(np.eye(3), dirv))
    axl.text(0, -0.07, "[001]", ha="center", fontsize=9)
    axl.text(1, -0.07, "[101]", ha="center", fontsize=9)
    axl.text(Cc[0] + 0.04, Cc[1], "[111]", va="center", fontsize=9)
    axl.set_title("IPF-Legende (Farbe = Kristallrichtung ∥ Aufbau)", fontsize=8.5)
    axl.set_aspect("equal"); axl.axis("off"); axl.set_xlim(-0.15, 1.25); axl.set_ylim(-0.2, 0.85)
    fig.tight_layout()
    out = os.path.join(HERE, "report", "fig_orientation_map.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out, "| section points:", m.sum())


if __name__ == "__main__":
    main()
