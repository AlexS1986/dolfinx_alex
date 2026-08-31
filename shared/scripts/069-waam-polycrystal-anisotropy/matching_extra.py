#!/usr/bin/env python3
"""
Extra matching diagnostics for 316L (report §3.2):
  (1) Texture matching  — {100} pole figures of the GENERATED RVE orientations
      vs the MEASURED EBSD (V-section, rotated into the RVE frame). Validates the
      area-weighted bootstrap and shows both are only weakly textured (cf. §3.6).
  (2) Grain-shape matching — per-grain aspect ratios of the generated RVE
      (from the mesh inertia tensor) vs the EBSD-derived target 3.41:3.18:1,
      plus the mean cell sphericity from the Neper .stcell.

Outputs (into report/):
  fig_texture_match.png, fig_shape_match.png   and prints a small summary table.

Usage:  python3 matching_extra.py [--neper-dir <neper_pipeline>] [--n 500]
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEPER = os.path.join(HERE, "..", "..", "..", "..", "Meshing", "Neper",
                             "data", "04_anisotropy_waam", "neper_pipeline")
NAVY = "#1F3864"; RED = "#c0392b"; BLUE = "#4c72b0"


def euler_to_g(phi1, Phi, phi2):
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([[c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s],
                     [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
                     [s1*s,             -c1*s,            c]])


def stereo(v):
    v = v * np.sign(v[2] + 1e-9)
    return v[0] / (1 + v[2]), v[1] / (1 + v[2])


def pf_points(gs, w):
    xs, ys, ss = [], [], []
    mw = np.mean(w)
    for g, wi in zip(gs, w):
        for col in g.T:               # <100> axes in sample coords
            x, y = stereo(col); xs.append(x); ys.append(y); ss.append(6 * wi / mw)
    return xs, ys, ss


def load_ebsd_vertical(neper):
    f = glob.glob(os.path.join(neper, "..", "data_c04", "*316L_Vertical*.txt"))[0]
    rows = []
    for ln in open(f, errors="ignore"):
        p = ln.split()
        if len(p) < 44:
            continue
        try:
            float(p[0])
        except ValueError:
            continue
        rows.append([float(x) for x in p[:44]])
    a = np.array(rows)
    a = a[(a[:, 30] >= 10) & (a[:, 36] > 0)]
    eul = np.deg2rad(a[:, [2, 3, 4]]); w = a[:, 31]
    # axial mean of ellipse angle -> theta (build in-plane axis of V map)
    th = np.deg2rad(a[:, 37]) * 2
    C = np.average(np.cos(th), weights=w); S = np.average(np.sin(th), weights=w)
    theta = np.arctan2(S, C) / 2
    return eul, w, theta


def grain_shape_from_msh(path):
    lines = open(path).read().split("\n")
    i = lines.index("$Nodes"); nn = int(lines[i + 1])
    nodes = {}
    for k in range(i + 2, i + 2 + nn):
        p = lines[k].split(); nodes[int(p[0])] = np.array([float(p[1]), float(p[2]), float(p[3])])
    j = lines.index("$Elements"); ne = int(lines[j + 1])
    V = defaultdict(float); M1 = defaultdict(lambda: np.zeros(3)); M2 = defaultdict(lambda: np.zeros((3, 3)))
    for k in range(j + 2, j + 2 + ne):
        p = lines[k].split()
        if len(p) < 3 or p[1] != '4':
            continue
        nt = int(p[2]); g = int(p[3]); nd = [int(x) for x in p[3 + nt:3 + nt + 4]]
        P = np.array([nodes[n] for n in nd]); c = P.mean(0)
        vol = abs(np.dot(P[1] - P[0], np.cross(P[2] - P[0], P[3] - P[0]))) / 6.0
        V[g] += vol; M1[g] += vol * c; M2[g] += vol * np.outer(c, c)
    bw, ww = [], []
    for g in V:
        mu = M1[g] / V[g]; Cov = M2[g] / V[g] - np.outer(mu, mu)
        d = np.sqrt(5 * np.clip(np.diag(Cov), 1e-12, None))  # semi-axes along x=weld,y=wall,z=build
        bw.append(d[2] / d[1]); ww.append(d[0] / d[1])
    return np.array(bw), np.array(ww)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neper-dir", default=DEFAULT_NEPER)
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()
    nd = args.neper_dir

    # ---- (1) texture: EBSD (rotated to RVE frame) vs generated ----
    eul, w, theta = load_ebsd_vertical(nd)
    th = theta
    Q = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1.0]])
    gs_ebsd = [euler_to_g(*e) @ Q for e in eul]
    eul_gen = np.loadtxt(os.path.join(nd, "grain_ori_316L.txt"),
                         comments="#", usecols=(1, 2, 3))
    if eul_gen.ndim == 1:
        eul_gen = eul_gen[None, :]
    gs_gen = [euler_to_g(*np.deg2rad(e)) for e in eul_gen]
    w_gen = np.ones(len(gs_gen))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5.2))
    for a, (gs, ww_, lab) in zip(ax, [(gs_ebsd, w, "gemessen (EBSD, V-Schliff)"),
                                      (gs_gen, w_gen, "generiert (RVE, n=%d)" % len(gs_gen))]):
        xs, ys, ss = pf_points(gs, ww_)
        a.scatter(xs, ys, s=ss, alpha=0.25, c=NAVY, edgecolors="none")
        t = np.linspace(0, 2 * np.pi, 200); a.plot(np.cos(t), np.sin(t), "k", lw=1.2)
        a.plot(0, 0, "r+", ms=11, mew=2)
        a.set_aspect("equal"); a.axis("off"); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.25, 1.12)
        a.set_title(lab, fontsize=11)
        a.text(0, -1.18, "Blick ∥ Aufbaurichtung (Zentrum)", ha="center", fontsize=8, color="0.4")
    fig.suptitle("{100}-Polfiguren: gemessene EBSD-Textur vs. generierte RVE-Textur (316L)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "report", "fig_texture_match.png"), dpi=150, bbox_inches="tight")
    print("wrote report/fig_texture_match.png")

    # ---- (2) grain shape: generated per-grain ratios vs target ----
    bw, ww = grain_shape_from_msh(os.path.join(nd, f"waam_316L_n{args.n}.msh"))
    sc = np.loadtxt(os.path.join(nd, f"waam_316L_n{args.n}.stcell"))
    sph = sc[:, 3].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(1, 9, 30)
    ax.hist(bw, bins=bins, alpha=0.55, color=BLUE, label="generiert: Aufbau/Wand")
    ax.hist(ww, bins=bins, alpha=0.55, color="#c0842f", label="generiert: Schweiß/Wand")
    ax.axvline(3.41, color=BLUE, ls="--", lw=2, label="Ziel Aufbau/Wand = 3.41")
    ax.axvline(3.18, color="#c0842f", ls="--", lw=2, label="Ziel Schweiß/Wand = 3.18")
    ax.set_xlabel("Korn-Aspektverhältnis (Halbachsen aus Trägheitstensor)")
    ax.set_ylabel("Anzahl Körner"); ax.legend(fontsize=8)
    ax.set_title(f"316L Kornform: generiert vs. EBSD-Ziel  (mittl. Sphärizität {sph:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "report", "fig_shape_match.png"), dpi=150, bbox_inches="tight")
    print("wrote report/fig_shape_match.png")

    print("\n=== shape summary (316L) ===")
    print(f"  Aufbau/Wand : Ziel 3.41 | generiert Median {np.median(bw):.2f}  Mittel {bw.mean():.2f}")
    print(f"  Schweiß/Wand: Ziel 3.18 | generiert Median {np.median(ww):.2f}  Mittel {ww.mean():.2f}")
    print(f"  mittlere Zell-Sphärizität (generiert): {sph:.2f}")


if __name__ == "__main__":
    main()
