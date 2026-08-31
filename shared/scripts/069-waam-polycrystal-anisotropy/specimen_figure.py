#!/usr/bin/env python3
"""
Compose the Neper renderings of the numerical tensile bars (directional
specimens V / H / 45deg, both steels) into one report figure. The per-bar
renderings are produced by the Neper pipeline (spec_<mat>_<orient>_view.png).

Output: report/fig_specimen_meshes.png

Usage:  python3 specimen_figure.py [--neper-dir <neper_pipeline>]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEPER = os.path.join(HERE, "..", "..", "..", "..", "Meshing", "Neper",
                             "data", "04_anisotropy_waam", "neper_pipeline")
ROWS = ["316L", "17-4PH"]
COLS = [("V", "V  (∥ Aufbau)"), ("H", "H  (⊥ Aufbau)"), ("45deg", "45°")]


def crop_white(img):
    """Crop surrounding white/transparent border."""
    a = img[..., :3] if img.shape[-1] == 4 else img
    mask = np.any(a < 0.96, axis=-1)
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    pad = 6
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, img.shape[1])
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neper-dir", default=DEFAULT_NEPER)
    args = ap.parse_args()

    fig, axs = plt.subplots(len(ROWS), len(COLS), figsize=(13, 6.2))
    for i, mat in enumerate(ROWS):
        for j, (orient, _) in enumerate(COLS):
            ax = axs[i, j]
            p = os.path.join(args.neper_dir, f"spec_{mat}_{orient}_view.png")
            if os.path.exists(p):
                im = crop_white(plt.imread(p))
                ax.imshow(im)
                H, W = im.shape[0], im.shape[1]
                # load-direction arrow (along the bar's long axis) below the bar
                ax.set_ylim(H * 1.20, -H * 0.03)
                y = H * 1.10
                ax.annotate("", xy=(W * 0.88, y), xytext=(W * 0.12, y),
                            arrowprops=dict(arrowstyle="<|-|>", color="#c0392b", lw=2.2))
                ax.text(W * 0.5, H * 1.17, "Zugrichtung $F$", ha="center", va="top",
                        color="#c0392b", fontsize=9)
            else:
                ax.text(0.5, 0.5, "(fehlt)", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(COLS[j][1], fontsize=11)
            if j == 0:
                ax.set_ylabel(mat, fontsize=12, rotation=90, labelpad=10)
    fig.suptitle("Numerische Zugstäbe (Neper-Tessellation, Körner eingefärbt): "
                 "gerichtete Proben V / H / 45° je Stahl", fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "report", "fig_specimen_meshes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
