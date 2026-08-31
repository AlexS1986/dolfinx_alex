#!/usr/bin/env python3
"""
{100} pole figures of the raw 316L EBSD sections (V / H / 45deg).

For every grain the three <100> crystal axes are projected stereographically
onto the section plane (viewing along the section normal = centre of the disc),
area-weighted. This visualises the crystallographic texture that the scan
actually carries: tight clustering = sharp texture, even spread = weak/random.

Used in report §3.6 to show that the available 316L scans are only weakly
textured (no sharp <100> cluster), consistent with the vs-experiment analysis.

Output: report/fig_polefigures_316L.png

Usage:  python3 pole_figures.py [--neper-dir <neper_pipeline>]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEPER = os.path.join(HERE, "..", "..", "..", "..", "Meshing", "Neper",
                             "data", "04_anisotropy_waam", "neper_pipeline")
N_PTS, AREA, PHI1, PHI, PHI2, MIN = 30, 31, 2, 3, 4, 36


def euler_to_g(phi1, Phi, phi2):
    c1, s1 = np.cos(phi1), np.sin(phi1)
    c, s = np.cos(Phi), np.sin(Phi)
    c2, s2 = np.cos(phi2), np.sin(phi2)
    return np.array([[c1*c2 - s1*s2*c,  s1*c2 + c1*s2*c, s2*s],
                     [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c, c2*s],
                     [s1*s,             -c1*s,            c]])


def load(neper, sub):
    f = glob.glob(os.path.join(neper, "..", "data_c04", f"*{sub}*.txt"))[0]
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
    return a[(a[:, N_PTS] >= 10) & (a[:, MIN] > 0)]


def stereo(v):
    v = v * np.sign(v[2] + 1e-9)            # upper hemisphere
    return v[0] / (1 + v[2]), v[1] / (1 + v[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neper-dir", default=DEFAULT_NEPER)
    args = ap.parse_args()

    panels = [("316L_Vertical", "V-Schliff  (Normale = Aufbaurichtung)"),
              ("316L_Horizontal", "H-Schliff  (Normale = Schweißrichtung)"),
              ("316L_45", "45°-Schliff")]
    fig, axs = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (sub, lab) in zip(axs, panels):
        d = load(args.neper_dir, sub); w = d[:, AREA]
        xs, ys, ss = [], [], []
        for r, wi in zip(d, w):
            g = euler_to_g(*np.deg2rad(r[[PHI1, PHI, PHI2]]))
            for col in g.T:                 # three <100> axes in sample coords
                x, y = stereo(col); xs.append(x); ys.append(y)
                ss.append(6 * wi / np.mean(w))
        ax.scatter(xs, ys, s=ss, alpha=0.25, c="#1f3864", edgecolors="none")
        t = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), "k", lw=1.2)
        ax.plot(0, 0, "r+", ms=12, mew=2)   # centre = section normal
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.28, 1.15)
        ax.set_title(lab, fontsize=10)
        ax.text(0, -1.20, "Mitte (+) = Schliffnormale   ·   Rand = in der Schliffebene",
                ha="center", fontsize=7, color="0.4")
    fig.suptitle("{100}-Polfiguren der 316L-Schliffe (EBSD-Rohdaten, flächengewichtet)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "report", "fig_polefigures_316L.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
