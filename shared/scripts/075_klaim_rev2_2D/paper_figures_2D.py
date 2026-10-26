#!/usr/bin/env python3
"""
Paper figures for Section 4 (2D) of the KLAIM 2025 revision 2, from the 075 results and evaluation_output/master_table.csv.

Writes pgf (vector, \\input in the manuscript) and png into ../046_plasticity/KLAIM_2025/revision_2/pics/ (names PAPER_rev2_*).
Usage: python3 evaluation_rev2.py ... (first, for master_table.csv)  then  python3 paper_figures_2D.py
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluation_rev2 as ev  # noqa: E402

RESULTS = os.path.join(HERE, "results")
TABLE = os.path.join(HERE, "evaluation_output", "master_table.csv")
OUT = os.path.join(HERE, "..", "046_plasticity", "KLAIM_2025", "revision_2", "pics")

plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "font.family": "serif", "font.size": 9,
                     "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "lines.linewidth": 1.0, "axes.grid": True, "grid.alpha": 0.3})
COL = {"vM": "#1b4f72", "EqJc": "#b03a2e", "EqSig": "#1e8449"}
MARK = {"vM": "o", "EqJc": "s", "EqSig": "^"}
LAB = {"vM": r"[vM]", "EqJc": r"[Eq $J_c$]", "EqSig": r"[Eq $\sigma^*$]"}
CASE = {("vM", 4.0): "S7_vM_ws4.0_mi5"}   # replacement run (CLAUDE.md 2026-10-21)
SIG_STAR_1D = {"vM": 1.28, "EqJc": 1.72, "EqSig": 1.28}   # Sec. 3.6, beta = 0.2


def case_name(sname, ws):
    return CASE.get((sname, ws), f"S1_{sname}_ws{ws:g}" if ws != int(ws) else f"S1_{sname}_ws{ws:.1f}")


def load(sname, ws):
    return ev.Case(os.path.join(RESULTS, case_name(sname, ws)))


def thin(x, y, n=1500):
    """keep the shape (incl. peaks) of long curves: per x-bin the first, min, max and last point"""
    if len(x) <= 2 * n:
        return x, y
    edges = np.linspace(x.min(), x.max(), n + 1)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, n - 1)
    keep = set()
    for b in np.unique(idx):
        sel = np.where(idx == b)[0]
        keep.update([sel[0], sel[-1], sel[np.argmin(y[sel])], sel[np.argmax(y[sel])]])
    k = np.array(sorted(keep))
    return x[k], y[k]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pgf"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    print("wrote", name)
    plt.close(fig)


def pores(ax, c, xlim=None):
    for xs, xe in zip(*ev.pore_positions(c)):
        if xlim is None or (xe > xlim[0] and xs < xlim[1]):
            ax.axvspan(xs, xe, color="0.92", zorder=0, lw=0)


def fig_xct_vs_t():
    c = load("vM", 1.0)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    t, x = thin(c.c("t"), c.x_ct())
    ax.plot(c.c("t"), c.c("x_bc"), color="0.4", ls="--", label=r"$x_{\mathrm{bc}}$")
    ax.plot(t, x, color=COL["vM"], label=r"$x_{\mathrm{ct}}$")
    ax.set_xlabel(r"Pseudo-time $t\,\dot{x}_{\mathrm{bc}}/L$"); ax.set_ylabel(r"Position $/L$")
    ax.legend(loc="upper left")
    fig.tight_layout(); save(fig, "PAPER_rev2_xct_xbc_vs_t_vM_ws1.0")


# --- J_x vs x_ct in the style of the v2.0 figures (markers only, pore edges dashed, legend outside), vector PDF incl. pore picture
OLD_COL = {"vM": "blue", "EqJc": "red", "EqSig": "green"}
OLD_MARK = {"vM": "o", "EqJc": "s", "EqSig": "D"}
WS_STYLE = [("blue", "o"), ("red", "s"), ("green", "D"), ("gray", "x"), ("orange", "^"), ("purple", "v"), ("brown", "<"),
            ("hotpink", ">"), ("cyan", "p"), ("magenta", "*"), ("goldenrod", "h")]
PORE_PIC = os.path.join(OUT, "holes_s_middle_row.png")   # s field of the v2.0 run, w_s = L, pores at x = 3, 5, ..., 13 L
PORE_PIC_EXTENT = (1.78, 14.32)                          # x range of the picture in units of L (pore centres at pixels 195 + 319 k of 2000)


def thin_markers(x, y, nx=700, ny=350):
    """one marker per (x, y) cell: keeps vertical streaks (trapped crack) and the ligament branches, drops duplicates"""
    if len(x) == 0:
        return x, y
    ix = np.floor((x - x.min()) / max(np.ptp(x), 1e-12) * nx).astype(int)
    iy = np.floor((y - y.min()) / max(np.ptp(y), 1e-12) * ny).astype(int)
    _, keep = np.unique(ix * (ny + 1) + iy, return_index=True)
    keep = np.sort(keep)
    return x[keep], y[keep]


def crack_tip(c):
    """running maximum of x_ct: the tracked node flickers by one element around crack jumps, the crack cannot heal"""
    return np.maximum.accumulate(c.x_ct())


def old_style(ax, c, xlim):
    starts, ends = ev.pore_positions(c)
    for xs, xe in zip(starts, ends):
        for xv in (xs, xe):
            if xlim[0] <= xv <= xlim[1]:
                ax.axvline(xv, color="k", ls="--", lw=0.5, zorder=0)
    ax.grid(False)


def fig_J_vs_xct_ws1():
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        xlim = (1.5, 14.5)
        fw, fh, left, right, top, bottom = 6.3, 4.3, 0.11, 0.78, 0.99, 0.12
        img = plt.imread(PORE_PIC)
        h = (PORE_PIC_EXTENT[1] - PORE_PIC_EXTENT[0]) * img.shape[0] / img.shape[1]
        h_pic = fw * (right - left) * h / (xlim[1] - xlim[0])          # strip height (inch) for round pores
        h_plot = fh * (top - bottom) - h_pic - 0.05
        fig = plt.figure(figsize=(fw, fh))
        gs = fig.add_gridspec(2, 1, height_ratios=[h_pic, h_plot], hspace=0.05 / ((h_pic + h_plot) / 2),
                              left=left, right=right, top=top, bottom=bottom)
        ax_pic = fig.add_subplot(gs[0]); ax = fig.add_subplot(gs[1], sharex=ax_pic)
        xlim = (1.5, 14.5)
        for s in ("vM", "EqJc", "EqSig"):
            c = load(s, 1.0)
            xa, Ja = crack_tip(c), c.c("Jx")
            sep = np.where((xa > 4.0) & (Ja < 0.1 * Ja.max()))[0]   # final separation of the specimen
            end = sep[0] if len(sep) else len(xa)
            x, J = thin_markers(xa[:end], Ja[:end])
            ax.plot(x, J, ls="none", marker=OLD_MARK[s], ms=1.6, color=OLD_COL[s], label=LAB[s])
        old_style(ax, c, xlim)
        ax_pic.imshow(img, extent=(PORE_PIC_EXTENT[0], PORE_PIC_EXTENT[1], 0, h), aspect="auto")
        ax_pic.set_ylim(0, h); ax_pic.axis("off")
        ax.set_xlim(*xlim); ax.set_ylim(0, 2.3); ax.set_xticks(np.arange(2, 15, 2))
        ax.set_xlabel(r"$x_{\mathrm{ct}}/L$"); ax.set_ylabel(r"$J_x / J_c^0$")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=4, frameon=True)
        fig.savefig(os.path.join(OUT, "PAPER_rev2_Jx_vs_xct_ws1.0_three_sets.pdf")); plt.close(fig)
        print("wrote PAPER_rev2_Jx_vs_xct_ws1.0_three_sets.pdf")


def fig_J_vs_xct_sections():
    """per parameter set: J_x vs x_ct between pores three and five (0-based 2..4), i.e. around the J_eff ligament, all w_s"""
    ws_list = [0.25, 0.375, 0.4, 0.5, 0.6, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0]
    for s in ("vM", "EqJc", "EqSig"):
        with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9, "xtick.labelsize": 10, "ytick.labelsize": 10}):
            fig, ax = plt.subplots(figsize=(6.3, 3.9))
            fig.subplots_adjust(left=0.11, right=0.78, top=0.97, bottom=0.14)
            for (col, mk), ws in zip(WS_STYLE, ws_list):
                c = load(s, ws)
                starts, ends = ev.pore_positions(c)
                xa, Ja = crack_tip(c), c.c("Jx")
                m = (xa >= starts[2] - 0.02) & (xa <= starts[4] + 0.02)
                if not np.any(m):
                    continue
                x, J = thin_markers(xa[m], Ja[m], 300, 250)
                ax.plot(x, J, ls="none", marker=mk, ms=1.8, color=col, label=rf"$w_s = {ws:g}L$")
            ax.set_ylim(bottom=0); ax.grid(False)
            ax.set_xlabel(r"$x_{\mathrm{ct}}/L$"); ax.set_ylabel(r"$J_x / J_c^0$")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), markerscale=3, frameon=True)
            name = f"PAPER_rev2_Jx_vs_xct_pores2to4_{s}"
            fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)
            print("wrote", name + ".pdf")


def fig_J_vs_xct_sections_combined():
    """Figs. 11-13 merged: one panel per parameter set, shared y axis, one legend (w_s) below"""
    ws_list = [0.25, 0.375, 0.4, 0.5, 0.6, 0.75, 1.0, 1.4, 2.0, 3.0, 4.0]
    with plt.rc_context({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True)
        fig.subplots_adjust(left=0.09, right=0.99, top=0.91, bottom=0.31, wspace=0.08)
        handles = []
        for ax, s, lab in zip(axes, ("vM", "EqJc", "EqSig"), ("a)", "b)", "c)")):
            for (col, mk), ws in zip(WS_STYLE, ws_list):
                c = load(s, ws)
                starts, _ = ev.pore_positions(c)
                xa, Ja = crack_tip(c), c.c("Jx")
                m = (xa >= starts[2] - 0.02) & (xa <= starts[4] + 0.02)
                if not np.any(m):
                    continue
                x, J = thin_markers(xa[m], Ja[m], 250, 220)
                h, = ax.plot(x, J, ls="none", marker=mk, ms=1.4, color=col, label=rf"$w_s = {ws:g}L$")
                if s == "vM":
                    handles.append(h)
            ax.grid(False); ax.set_ylim(0, 2.35)
            ax.set_xlabel(r"$x_{\mathrm{ct}}/L$")
            ax.set_title(f"{lab} {LAB[s]}", fontsize=10)
        axes[0].set_ylabel(r"$J_x / J_c^0$")
        fig.legend(handles=handles, loc="lower center", ncol=6, markerscale=3, frameon=False, columnspacing=1.0,
                   handletextpad=0.3, bbox_to_anchor=(0.54, 0.0))
        fig.savefig(os.path.join(OUT, "PAPER_rev2_Jx_vs_xct_sections_three_sets.pdf")); plt.close(fig)
        print("wrote PAPER_rev2_Jx_vs_xct_sections_three_sets.pdf")


def fig_references():
    """Fig. 3 (Sec. 3.5): J_x vs x_ct in the homogeneous references (16L x 20L box without pores, K_bc^2 = 2.5): [vM] solid
    (yield stress everywhere), [Eq J_c] and [Eq sigma*] homogeneous elastic; plateau window 0.3-0.7 max(x_bc) shaded. Solver output only."""
    refs = (("vM", "S2_solid_vM"), ("EqJc", "S2_strip_EqJc"), ("EqSig", "S2_strip_EqSig"))
    with plt.rc_context({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 9.5, "xtick.labelsize": 10, "ytick.labelsize": 10}):
        fig, ax = plt.subplots(figsize=(6.3, 3.6))
        lo = hi = None
        for s, name in refs:
            c = ev.Case(os.path.join(RESULTS, name))
            jc = ev.Jc_num_from_strip(c)
            x, J = thin(crack_tip(c), c.c("Jx"), 1500)
            ax.plot(x, J, color=OLD_COL[s], lw=1.2, label=LAB[s] + rf" ($J_c^{{\mathrm{{num}}}} = {jc:.2f}\,J_c^0$)")
            xmax = np.max(c.c("x_bc"))
            lo, hi = 0.3 * xmax, 0.7 * xmax
        ax.axvspan(lo, hi, color="0.93", zorder=0, lw=0)
        ax.set_xlim(1.5, 16); ax.set_ylim(0, 1.9); ax.grid(False)
        ax.set_xlabel(r"$x_{\mathrm{ct}}/L$"); ax.set_ylabel(r"$J_x / J_c^0$")
        ax.legend(loc="lower center", frameon=True)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "PAPER_rev2_Jx_vs_xct_homogeneous_references.pdf")); plt.close(fig)
        print("wrote PAPER_rev2_Jx_vs_xct_homogeneous_references.pdf")


def rows():
    out = [r for r in csv.DictReader(open(TABLE)) if r["baseline"] == "True" and r["dim"] == "2"]
    return {(r["set"], float(r["wsteg"])): r for r in out}


def fig_Jeff_vs_ws(R):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for s in ("vM", "EqJc", "EqSig"):
        ws = sorted(w for (ss, w) in R if ss == s)
        axes[0].plot(ws, [float(R[(s, w)]["Jeff"]) for w in ws], color=COL[s], marker=MARK[s], ms=3.5, label=LAB[s])
        axes[1].plot(ws, [float(R[(s, w)]["Jeff_over_Jcnum"]) for w in ws], color=COL[s], marker=MARK[s], ms=3.5, label=LAB[s])
    axes[0].set_ylabel(r"$J_{x,c}^{\mathrm{eff}} / J_c^0$"); axes[1].set_ylabel(r"$J_{x,c}^{\mathrm{eff}} / J_c^{\mathrm{num}}$")
    for ax, lab in zip(axes, ("a)", "b)")):
        ax.set_xlabel(r"Wall width $w_s / L$"); ax.set_ylim(bottom=0); ax.text(0.02, 0.95, lab, transform=ax.transAxes, va="top")
    axes[1].legend(loc="lower right")
    fig.tight_layout(); save(fig, "PAPER_rev2_Jeff_vs_wsteg_three_sets")


def fig_energy_partition(R):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for s in ("vM", "EqSig", "EqJc"):
        ws = sorted(w for (ss, w) in R if ss == s)
        f = lambda k: [float(R[(s, w)][k]) for w in ws]
        if s != "EqJc":   # EqJc = EqSig (self-similar), only in panel b
            axes[0].plot(ws, f("cyc_fracture"), color=COL[s], marker="o", ms=3, label=LAB[s] + r": fracture")
            axes[0].plot(ws, f("cyc_visc"), color=COL[s], marker="s", ms=3, ls="--", label=LAB[s] + r": viscous")
            if s == "vM":
                axes[0].plot(ws, f("cyc_plastic"), color=COL[s], marker="^", ms=3, ls=":", label=LAB[s] + r": plastic")
        axes[1].plot(ws, [g / j for g, j in zip(f("cyc_G_diss"), f("Jeff"))], color=COL[s], marker=MARK[s], ms=3.5, label=LAB[s])
    axes[0].set_ylabel(r"Fraction of $\Delta D_{\mathrm{tot}}$ per cycle"); axes[0].set_ylim(0, 1)
    axes[1].set_ylabel(r"$G_{\mathrm{diss}} / J_{x,c}^{\mathrm{eff}}$"); axes[1].set_ylim(0, 1.3)
    for ax, lab in zip(axes, ("a)", "b)")):
        ax.set_xlabel(r"Wall width $w_s / L$"); ax.text(0.02, 0.95, lab, transform=ax.transAxes, va="top")
    axes[0].legend(loc="center right", fontsize=6.3); axes[1].legend(loc="lower left")
    fig.tight_layout(); save(fig, "PAPER_rev2_energy_partition_cycle")


def ligament_state(c):
    starts, _ = ev.pore_positions(c)
    xh, x = starts[3], c.x_ct()
    m = np.where((x >= xh - 1.01 * c.wsteg) & (x <= xh + 0.01 * c.wsteg))[0]
    k = m[np.argmax(c.c("Jx")[m])]
    return {q: float(c.c(q)[k]) for q in ("sig_yy_max_lig", "sig_vM_max_lig", "sig_h_max_lig", "alpha_max_lig")}


def fig_ligament_state(R):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for s in ("vM", "EqJc", "EqSig"):
        ws = sorted(w for (ss, w) in R if ss == s)
        st = [ligament_state(load(s, w)) for w in ws]
        axes[0].plot(ws, [d["sig_yy_max_lig"] / SIG_STAR_1D[s] for d in st], color=COL[s], marker=MARK[s], ms=3.5, label=LAB[s])
        axes[1].plot(ws, [d["sig_h_max_lig"] / d["sig_vM_max_lig"] for d in st], color=COL[s], marker=MARK[s], ms=3.5, label=LAB[s])
        if s == "vM":
            ax2 = axes[1].twinx()
            ax2.plot(ws, [d["alpha_max_lig"] for d in st], color=COL[s], ls=":", marker="x", ms=3.5)
            ax2.set_ylabel(r"$\alpha_{\max}$ (dotted, [vM])"); ax2.set_ylim(0, 1); ax2.grid(False)
    axes[0].axhline(1.0, color="0.5", lw=0.8)
    axes[0].set_ylabel(r"$\sigma_{yy,\max} / \sigma^*_{\mathrm{1D}}$"); axes[0].set_ylim(0.8, 1.45)
    axes[1].set_ylabel(r"$\sigma_h / \sigma_{\mathrm{vM}}$ at $\max$"); axes[1].set_ylim(0.3, 1.2)
    for ax, lab in zip(axes, ("a)", "b)")):
        ax.set_xlabel(r"Wall width $w_s / L$"); ax.text(0.02, 0.95, lab, transform=ax.transAxes, va="top")
    axes[0].legend(loc="lower right", fontsize=6.5)
    fig.tight_layout(); save(fig, "PAPER_rev2_ligament_state_before_renucleation")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    R = rows()
    fig_xct_vs_t()
    fig_J_vs_xct_ws1()
    fig_J_vs_xct_sections()
    fig_J_vs_xct_sections_combined()
    fig_references()
    fig_Jeff_vs_ws(R)
    fig_energy_partition(R)
    fig_ligament_state(R)
