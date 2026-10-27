#!/usr/bin/env python3
"""
Paper figures for Section 5 (3D) of the KLAIM 2025 revision 2: fixed-mesh runs S1F (h_band = 0.1L), normalised by the solid
reference on the same mesh (S2F), mesh-convergence pair S9F (h_band = 0.07L), comparison with the 2D runs of 075.
Same style and helpers as ../075_klaim_rev2_2D/paper_figures_2D.py. Output: ../046_plasticity/KLAIM_2025/revision_2/pics/PAPER_rev2_3D_*
Usage: python3 ../075_klaim_rev2_2D/evaluation_rev2.py --results ../075_klaim_rev2_2D/results --results3d results  (first)
       python3 paper_figures_3D.py
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "075_klaim_rev2_2D"))
import paper_figures_2D as pf  # noqa: E402  (sets the pgf backend and rcParams)
import evaluation_rev2 as ev  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

R3 = os.path.join(HERE, "results")
R2 = os.path.join(HERE, "..", "075_klaim_rev2_2D", "results")
TABLE = os.path.join(HERE, "..", "075_klaim_rev2_2D", "evaluation_output", "master_table.csv")
OUT = pf.OUT
WS3 = [0.375, 0.5, 0.75, 1.0, 2.0]
COL3D, MARK3D = "purple", "^"


def case3(ws, series="S1G"):
    """3D porous run: S1G = 6x3x2 series (paper, 2026-10-26), S1F = 4x3x2 check series"""
    return ev.Case(os.path.join(R3, f"{series}_3D_vM_ws{ws:g}" if ws != int(ws) else f"{series}_3D_vM_ws{ws:.1f}"))


def rows():
    return {r["name"]: r for r in csv.DictReader(open(TABLE))}


def jc_ref(R, name):
    return float(R[name]["Jc_ref_value"])


def fig_xct_vs_t_3D():
    c = case3(1.0)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    t, x = pf.thin(c.c("t"), pf.crack_tip(c))
    ax.plot(c.c("t"), c.c("x_bc") + c.x_offset, color="0.4", ls="--", label=r"$x_{\mathrm{bc}}$")
    ax.plot(t, x, color=COL3D, label=r"$x_{\mathrm{ct}}$")
    ax.set_xlabel(r"Pseudo-time $t\,\dot{x}_{\mathrm{bc}}/L$"); ax.set_ylabel(r"Position $/L$")
    ax.legend(loc="upper left")
    fig.tight_layout(); pf.save(fig, "PAPER_rev2_3D_xct_xbc_vs_t_ws1.0")


def fig_J_vs_xct_2D_vs_3D(R):
    """w_s = L, J_x / J_c^num of the own dimension (2D solid 1.425, 3D solid same mesh), markers, pore strip on top"""
    c2 = ev.Case(os.path.join(R2, "S1_vM_ws1.0")); c3 = case3(1.0)
    jc2, jc3 = jc_ref(R, "S1_vM_ws1.0"), jc_ref(R, c3.name)
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        xlim = (1.5, 15.5)
        fw, fh, left, right, top, bottom = 6.3, 4.0, 0.11, 0.78, 0.99, 0.13
        img = plt.imread(pf.PORE_PIC)
        h = (pf.PORE_PIC_EXTENT[1] - pf.PORE_PIC_EXTENT[0]) * img.shape[0] / img.shape[1]
        h_pic = fw * (right - left) * h / (xlim[1] - xlim[0])
        h_plot = fh * (top - bottom) - h_pic - 0.05
        fig = plt.figure(figsize=(fw, fh))
        gs = fig.add_gridspec(2, 1, height_ratios=[h_pic, h_plot], hspace=0.05 / ((h_pic + h_plot) / 2),
                              left=left, right=right, top=top, bottom=bottom)
        ax_pic = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=ax_pic)
        for c, jc, col, mk, lab in ((c2, jc2, "blue", "o", "2D"), (c3, jc3, COL3D, MARK3D, "3D")):
            xa, Ja = pf.crack_tip(c), c.c("Jx") / jc
            sep = np.where((xa > 4.0) & (Ja < 0.1 * Ja.max()))[0]
            end = sep[0] if len(sep) else len(xa)
            m = xa[:end] <= xlim[1]
            x, J = pf.thin_markers(xa[:end][m], Ja[:end][m])
            ax.plot(x, J, ls="none", marker=mk, ms=1.6, color=col, label=lab)
        pf.old_style(ax, c3, xlim)
        ax_pic.imshow(img, extent=(pf.PORE_PIC_EXTENT[0], pf.PORE_PIC_EXTENT[1], 0, h), aspect="auto")
        ax_pic.set_ylim(0, h); ax_pic.axis("off")
        ax.set_xlim(*xlim); ax.set_ylim(0, 1.3); ax.set_xticks(np.arange(2, 16, 2))
        ax.set_xlabel(r"$x_{\mathrm{ct}}/L$"); ax.set_ylabel(r"$J_x / J_c^{\mathrm{num}}$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=4, frameon=True)
        fig.savefig(os.path.join(OUT, "PAPER_rev2_3D_Jx_vs_xct_2D_vs_3D_ws1.0.pdf")); plt.close(fig)
        print("wrote PAPER_rev2_3D_Jx_vs_xct_2D_vs_3D_ws1.0.pdf")


def fig_J_vs_xct_sections_3D(R):
    """3D, J_x / J_c^num vs x_ct around the evaluation ligament (between pore starts 1 and 3, 0-based), all fixed-mesh w_s"""
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.3, 3.6))
        fig.subplots_adjust(left=0.11, right=0.78, top=0.97, bottom=0.15)
        styles = [pf.WS_STYLE[i] for i in (1, 2, 5, 6, 8)]   # same colours/markers per w_s as in the 2D Fig. 11
        for (col, mk), ws in zip(styles, WS3):
            try:
                c = case3(ws)
                jc = jc_ref(R, c.name)
            except (KeyError, ValueError, OSError):
                continue
            starts, _ = ev.pore_positions(c)
            k = ev.eval_cycle(c)                      # around the evaluated ligament (cycle 2 for six pores)
            xa, Ja = pf.crack_tip(c), c.c("Jx") / jc
            m = (xa >= starts[k] - 0.02) & (xa <= starts[k + 2] + 0.02)
            if not np.any(m):
                continue
            x, J = pf.thin_markers(xa[m], Ja[m], 300, 250)
            ax.plot(x, J, ls="none", marker=mk, ms=1.8, color=col, label=rf"$w_s = {ws:g}L$")
        ax.set_ylim(0, 1.2); ax.grid(False)
        ax.set_xlabel(r"$x_{\mathrm{ct}}/L$"); ax.set_ylabel(r"$J_x / J_c^{\mathrm{num}}$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=3, frameon=True)
        fig.savefig(os.path.join(OUT, "PAPER_rev2_3D_Jx_vs_xct_sections.pdf")); plt.close(fig)
        print("wrote PAPER_rev2_3D_Jx_vs_xct_sections.pdf")


def fig_Jeff_2D_vs_3D(R):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    r2 = sorted([r for r in R.values() if r["baseline"] == "True" and r["dim"] == "2" and r["set"] == "vM"], key=lambda r: float(r["wsteg"]))
    r3 = sorted([r for r in R.values() if r["dim"] == "3" and r["baseline"] == "True" and "_mi" not in r["name"]
                 and np.isfinite(float(r["Jeff_over_Jcnum"] or "nan"))], key=lambda r: float(r["wsteg"]))
    r4 = sorted([r for r in R.values() if r["name"].startswith("S1F_3D_vM_ws") and "_mi" not in r["name"]], key=lambda r: float(r["wsteg"]))
    f = lambda rr, k: [float(r[k]) for r in rr]
    axes[0].plot(f(r2, "wsteg"), f(r2, "Jeff_over_Jcnum"), color="blue", marker="o", ms=3.5, label=r"2D, [vM]")
    axes[0].plot(f(r3, "wsteg"), f(r3, "Jeff_over_Jcnum"), color=COL3D, marker=MARK3D, ms=4, label=r"3D, [vM], $6\times3\times2$")
    axes[0].plot(f(r4, "wsteg"), f(r4, "Jeff_over_Jcnum"), color=COL3D, marker="v", ms=4, mfc="none", ls=":", lw=0.8,
                 label=r"3D, [vM], $4\times3\times2$")
    r9 = R.get("S9F_3D_vM_ws1.0_h07")
    if r9:
        axes[0].plot([1.0], [float(r9["Jeff_over_Jcnum"])], ls="none", marker="s", ms=5, mfc="none", color=COL3D,
                     label=r"3D, $4\times3\times2$, $h = 0.07L$")
    axes[0].axhline(1.0, color="0.5", lw=0.8)
    axes[0].set_ylabel(r"$J_{x,c}^{\mathrm{eff}} / J_c^{\mathrm{num}}$"); axes[0].set_ylim(0, 1.45)
    axes[1].plot(f(r2, "wsteg"), f(r2, "cyc_visc"), color="blue", marker="s", ms=3, ls="--", label="2D viscous")
    axes[1].plot(f(r2, "wsteg"), f(r2, "cyc_fracture"), color="blue", marker="o", ms=3, label="2D fracture")
    axes[1].plot(f(r3, "wsteg"), f(r3, "cyc_visc"), color=COL3D, marker="s", ms=3, ls="--", label="3D viscous")
    axes[1].plot(f(r3, "wsteg"), f(r3, "cyc_fracture"), color=COL3D, marker=MARK3D, ms=3.5, label="3D fracture")
    axes[1].set_ylabel(r"Fraction of $\Delta D_{\mathrm{tot}}$ per cycle"); axes[1].set_ylim(0, 1)
    for ax, lab, yl in zip(axes, ("a)", "b)"), (0.95, 0.12)):
        ax.set_xlabel(r"Wall width $w_s / L$"); ax.text(0.02, yl, lab, transform=ax.transAxes, va="top")
    axes[0].legend(loc="lower right", fontsize=6.5); axes[1].legend(loc="center right", fontsize=6.5)
    fig.tight_layout(); pf.save(fig, "PAPER_rev2_3D_Jeff_and_partition_2D_vs_3D")


if __name__ == "__main__":
    R = rows()
    fig_xct_vs_t_3D()
    fig_J_vs_xct_2D_vs_3D(R)
    fig_J_vs_xct_sections_3D(R)
    fig_Jeff_2D_vs_3D(R)
