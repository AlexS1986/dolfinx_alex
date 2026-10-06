#!/usr/bin/env python3
"""Manuscript figures and tables for the M = 1e5 campaign (260930_M1e5), revision of A03.

Replaces the fracture-result figures of the submitted manuscript (M = 100, traction-based left-strip reaction)
by energy-consistent, first-peak based versions computed from the residual columns 14-17 of the campaign
histories. Reuses the loaders and the peak definition of 14_energy_consistent_metrics.py.

Figures (plots/energy_consistent/manuscript/, pdf + png):
  Response_energy_grid_vs_uy_M1e5_eps0_03      (a) total R_y, (b) W, (c) Pi_el/Pi_frac/D_s, (d) W vs Pi_tot, to u_y = 0.03 mm
  Peak_metrics_grid_vs_beta_s_M1e5             (a) R_y^(1), (b) W^(1), (c) Pi_el^(1), (d) Pi_frac^(1) vs beta_s
  First_peak_Ry_vs_sigma_c_M1e5                R_y^(1) vs effective sigma_c (Gc, mu averages from summary_260504.csv)
  Beta_comparison_Ry_vs_uy_M1e5_eps0_03        E_var curves for the three beta_phi, both rho
  Beta_comparison_first_peak_vs_beta_s_M1e5    E_var first peak vs beta_s for the three beta_phi (+ E_max reference)
  Mobility_convergence_first_peak              (a) R^(1)(M)/R^(1)(1e5) vs M with M^(-2/3) fits, (b) D_s/W at first peak vs M
Tables (LaTeX bodies + CSV): first-peak measures, secondary measures, dissipation ratios, comparisons, beta_phi study,
energy balance at the first peak and at the end.

Usage:
  python3 15_manuscript_figures_M1e5.py [--campaign-root results/campaign_260930_M1e5] [--reference-root results/new_W_whole_boundary]
      [--mobility-root DIR ...] [--summary-csv 68c3b8d0b7dca7b64b8b7a93/Images_A02/summary_260504.csv] [--output-dir DIR]
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ec", SCRIPT_DIR / "14_energy_consistent_metrics.py")
ec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ec)

CAMPAIGN_MOBILITY = 1.0e5
END_DISPLACEMENT = 0.03
PANEL_LABEL_SIZE = 26
RHO_COLORS = {0.3: "#c43c3c", 0.6: "#2f6fb7"}
CASE_LINESTYLES = {"min": "-", "max": (0, (6.0, 3.0)), "vary": (0, (5.0, 2.0, 1.0, 2.0))}
BETA_PHI_LINESTYLES = {0.001: (0, (1.2, 2.2)), 0.01: (0, (6.0, 3.0)), 0.05: "-"}
BETA_PHI_MARKERS = {0.001: "^", 0.01: "o", 0.05: "s"}
CASE_ORDER = ["min", "max", "vary"]
STRUCTURES = ec.STRUCTURES
COLORS = ec.ENERGY_DATASET_COLORS
CASE_LABELS = ec.CASE_LABELS
CASE_TEXT = ec.CASE_TEXT
COMPARISONS = ec.COMPARISONS
LEGEND_FONT_SIZE = 20

_trapz = ec._trapz


# ----------------------------------------------------------------------------------------------- data

def load_history(path):
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    return data


def campaign_record(path):
    """Full record of a residual-column run (campaign): curves and first-peak / end-state measures."""
    data = load_history(path)
    if data.shape[1] <= ec.COL_W_RES:
        raise ValueError(f"{path} has no residual columns")
    u = np.abs(data[:, ec.COL_UY])
    r = np.abs(data[:, ec.COL_RES_L] + data[:, ec.COL_RES_R])
    w = np.abs(data[:, ec.COL_W_RES])
    pi_el, pi_frac, d_s = data[:, ec.COL_PI_EL], data[:, ec.COL_PI_FRAC], data[:, ec.COL_D_S]
    pi_tot = pi_el + pi_frac + d_s
    r_tr = np.abs(data[:, ec.COL_RY_L] + data[:, ec.COL_RY_R])
    p = ec.first_peak_index(r)
    p_tr = ec.first_peak_index(r_tr)
    imax = int(np.argmax(r))
    last = len(u) - 1
    return {
        "u": u, "R": r, "W": w, "Pi_el": pi_el, "Pi_frac": pi_frac, "D_s": d_s, "Pi_tot": pi_tot,
        "peak": p,
        "R1": r[p], "u1": u[p], "W1": w[p], "Pi_el1": pi_el[p], "Pi_frac1": pi_frac[p], "D_s1": d_s[p],
        "balance1": w[p] / pi_tot[p],
        "Rmax": r[imax], "u_Rmax": u[imax], "Rmax_over_R1": r[imax] / r[p], "ended_rising": bool(imax == last and last > p),
        "u_end": u[-1], "W_end": w[-1], "Pi_el_end": pi_el[-1], "Pi_frac_end": pi_frac[-1], "D_s_end": d_s[-1],
        "balance_end": w[-1] / pi_tot[-1],
        "Ds_over_W1": d_s[p] / w[p], "Ds_share1": d_s[p] / (pi_frac[p] + d_s[p]) if pi_frac[p] + d_s[p] > 0 else np.nan,
        "Ds_over_W_end": d_s[-1] / w[-1], "Ds_share_end": d_s[-1] / (pi_frac[-1] + d_s[-1]),
        "R1_traction": r_tr[p_tr], "Ds_over_W1_traction": d_s[p_tr] / pi_tot[p_tr],
    }


def traction_first_peak(path):
    """First peak of the traction-based total reaction and D_s/W_exact there (available for every M)."""
    data = load_history(path)
    r_tr = np.abs(data[:, ec.COL_RY_L] + data[:, ec.COL_RY_R])
    pi_tot = data[:, ec.COL_PI_EL] + data[:, ec.COL_PI_FRAC] + data[:, ec.COL_D_S]
    p = ec.first_peak_index(r_tr)
    return r_tr[p], data[p, ec.COL_D_S] / pi_tot[p], np.abs(data[p, ec.COL_UY])


def load_sigma_c(summary_csv):
    """Effective sigma_c per (beta_phi, rho, case, beta_s) from the submission summary (material fields unchanged)."""
    out = {}
    with open(summary_csv) as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "spectral" or int(row["a"]) != 6:
                continue
            case = {"min": "min", "max": "max", "var": "vary", "vary": "vary"}[row["case"]]
            out[(float(row["beta"]), float(row["rho"]), case, float(row["epsilon"]))] = (
                float(row["sigma_c"]), float(row["Gc_average"]), float(row["mu_average"]))
    return out


# ----------------------------------------------------------------------------------------------- style

def style(ax):
    ax.tick_params(axis="both", labelsize=ec.TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(ec.AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(ec.AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3)


def panel_label(ax, text):
    ax.text(0.03, 0.95, rf"\textbf{{({text})}}", transform=ax.transAxes, ha="left", va="top", fontsize=PANEL_LABEL_SIZE)


def save(fig, output_dir, stem):
    legends = [child for ax in fig.axes for child in ax.get_children() if isinstance(child, matplotlib.legend.Legend)]
    legends += [child for child in fig.get_children() if isinstance(child, matplotlib.legend.Legend)]
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)
    print(f"[INFO] wrote {stem}")


def structure_handles_energy():
    return [Line2D([0], [0], color=COLORS[(rho, case)], linewidth=2.4, label=rf"$\rho={rho:g}$, {CASE_LABELS[case]}")
            for rho, case in STRUCTURES]


def structure_handles_rho(marker="o"):
    handles = []
    for rho in (0.3, 0.6):
        for case in CASE_ORDER:
            if (rho, case) not in STRUCTURES:
                continue
            handles.append(Line2D([0], [0], color=RHO_COLORS[rho], linestyle=CASE_LINESTYLES[case], marker=marker,
                                  markersize=7, linewidth=2.2, label=rf"$\rho={rho:g}$, {CASE_LABELS[case]}"))
    return handles


def first_peak_handle():
    return Line2D([0], [0], marker="X", color="#2f2f2f", linewidth=0.0, markersize=11, markeredgecolor="#1f1f1f",
                  label=r"first peak $R_y^{(1)}$")


# ----------------------------------------------------------------------------------------------- figures

def fig_response_energy_grid(recs, beta_phi, epsilon, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 12.5))
    (ax_r, ax_w), (ax_e, ax_b) = axes
    for rho, case in STRUCTURES:
        rec = recs.get((beta_phi, epsilon, rho, case))
        if rec is None:
            continue
        c = COLORS[(rho, case)]
        u, p = rec["u"], rec["peak"]
        mk = dict(s=110, marker="X", color=c, edgecolor="#1f1f1f", linewidth=0.85, zorder=6)
        ax_r.plot(u, rec["R"], color=c, linewidth=2.2)
        ax_r.scatter(u[p], rec["R"][p], **mk)
        ax_w.plot(u, rec["W"], color=c, linewidth=2.2)
        ax_w.scatter(u[p], rec["W"][p], **mk)
        ax_e.plot(u, rec["Pi_el"], color=c, linewidth=2.2)
        ax_e.plot(u, rec["Pi_frac"], color=c, linewidth=2.0, linestyle=(0, (6.0, 3.0)))
        ax_e.plot(u, rec["D_s"], color=c, linewidth=1.8, linestyle=(0, (1.2, 2.2)))
        ax_e.scatter(u[p], rec["Pi_el"][p], **mk)
        ax_b.plot(u, rec["W"], color=c, linewidth=2.2)
        ax_b.plot(u, rec["Pi_tot"], color=c, linewidth=2.0, linestyle=(0, (6.0, 3.0)))
        ax_b.scatter(u[p], rec["W"][p], **mk)
    for ax, label, ylabel in ((ax_r, "a", r"total $R_y$ in N/mm"), (ax_w, "b", r"$W$ in Nmm/mm"),
                              (ax_e, "c", r"$\Pi_\mathrm{el}$, $\Pi_\mathrm{frac}$, $D_s$ in Nmm/mm"),
                              (ax_b, "d", r"$W$, $\Pi_\mathrm{tot}$ in Nmm/mm")):
        ax.set_xlim(0.0, END_DISPLACEMENT)
        ax.set_ylim(0.0, 1.12 * ax.get_ylim()[1])
        ax.set_ylabel(ylabel)
        style(ax)
        panel_label(ax, label)
    ax_e.set_ylim(0.0, 3.0)  # Pi_frac and D_s visible; the largest Pi_el curves are clipped
    for ax in (ax_e, ax_b):
        ax.set_xlabel(r"$u_y$ in mm")
    for ax in (ax_r, ax_w):
        ax.tick_params(labelbottom=False)
    ax_e.legend(handles=[Line2D([0], [0], color="#2f2f2f", linewidth=2.2, label=r"$\Pi_\mathrm{el}$"),
                         Line2D([0], [0], color="#2f2f2f", linewidth=2.0, linestyle=(0, (6.0, 3.0)), label=r"$\Pi_\mathrm{frac}$"),
                         Line2D([0], [0], color="#2f2f2f", linewidth=1.8, linestyle=(0, (1.2, 2.2)), label=r"$D_s$")],
                fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    ax_b.legend(handles=[Line2D([0], [0], color="#2f2f2f", linewidth=2.2, label=r"$W$"),
                         Line2D([0], [0], color="#2f2f2f", linewidth=2.0, linestyle=(0, (6.0, 3.0)), label=r"$\Pi_\mathrm{tot}$")],
                fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(0.02, 0.88))
    fig.legend(handles=structure_handles_energy(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=LEGEND_FONT_SIZE, frameon=True)
    fig.legend(handles=[first_peak_handle()], loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1,
               fontsize=LEGEND_FONT_SIZE, frameon=True)
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    save(fig, output_dir, f"Response_energy_grid_vs_uy_M1e5_eps{str(epsilon).replace('.', '_')}")


def fig_peak_metrics_grid(recs, beta_phi, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 12.5))
    panels = [("a", "R1", r"$R_y^{(1)}$ in N/mm"), ("b", "W1", r"$W^{(1)}$ in Nmm/mm"),
              ("c", "Pi_el1", r"$\Pi_\mathrm{el}^{(1)}$ in Nmm/mm"), ("d", "Pi_frac1", r"$\Pi_\mathrm{frac}^{(1)}$ in Nmm/mm")]
    for ax, (label, key, ylabel) in zip(axes.flat, panels):
        for rho, case in STRUCTURES:
            eps = sorted(e for (b, e, r, c) in recs if b == beta_phi and r == rho and c == case)
            if not eps:
                continue
            ax.plot(eps, [recs[(beta_phi, e, rho, case)][key] for e in eps], color=RHO_COLORS[rho],
                    linestyle=CASE_LINESTYLES[case], marker="o", markersize=8, linewidth=2.2)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.15 * ax.get_ylim()[1])
        ax.set_xlim(0.012, 0.063)
        style(ax)
        panel_label(ax, label)
    for ax in axes[1]:
        ax.set_xlabel(r"$\beta_s$ in mm")
    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    fig.legend(handles=structure_handles_rho(), loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=LEGEND_FONT_SIZE, frameon=True)
    fig.tight_layout(h_pad=1.5, w_pad=2.0)
    save(fig, output_dir, "Peak_metrics_grid_vs_beta_s_M1e5")


def fig_first_peak_vs_sigma_c(recs, sigma, beta_phi, output_dir):
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for rho, case in STRUCTURES:
        eps = sorted(e for (b, e, r, c) in recs if b == beta_phi and r == rho and c == case)
        pts = [(sigma[(beta_phi, rho, case, e)][0], recs[(beta_phi, e, rho, case)]["R1"]) for e in eps
               if (beta_phi, rho, case, e) in sigma]
        if not pts:
            continue
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=RHO_COLORS[rho], linestyle=CASE_LINESTYLES[case],
                marker="o", markersize=8, linewidth=2.2)
    ax.set_xlabel(r"$\sigma_c$ in N/mm$^2$")
    ax.set_ylabel(r"$R_y^{(1)}$ in N/mm")
    ax.set_ylim(bottom=0.0)
    style(ax)
    ax.legend(handles=structure_handles_rho(), fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    save(fig, output_dir, "First_peak_Ry_vs_sigma_c_M1e5")


def fig_beta_comparison_curves(recs, epsilon, output_dir):
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for rho in (0.3, 0.6):
        for beta_phi, ls in BETA_PHI_LINESTYLES.items():
            rec = recs.get((beta_phi, epsilon, rho, "vary"))
            if rec is None:
                continue
            u, p = rec["u"], rec["peak"]
            ax.plot(u, rec["R"], color=RHO_COLORS[rho], linestyle=ls, linewidth=2.0)
            ax.scatter(u[p], rec["R"][p], s=110, marker="X", color=RHO_COLORS[rho], edgecolor="#1f1f1f", linewidth=0.85, zorder=6)
    ax.set_xlim(0.0, END_DISPLACEMENT)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(r"total $R_y$ in N/mm")
    style(ax)
    beta_handles = [Line2D([0], [0], color="#2f2f2f", linestyle=ls, linewidth=2.0, label=rf"$\beta_\phi={b:g}\,\mathrm{{mm^2}}$")
                    for b, ls in BETA_PHI_LINESTYLES.items()]
    rho_handles = [Line2D([0], [0], color=RHO_COLORS[rho], linewidth=2.4, label=rf"$\rho={rho:g}$") for rho in (0.3, 0.6)]
    first = ax.legend(handles=beta_handles, fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=rho_handles + [first_peak_handle()], fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save(fig, output_dir, f"Beta_comparison_Ry_vs_uy_M1e5_eps{str(epsilon).replace('.', '_')}")


def fig_beta_comparison_first_peak(recs, output_dir):
    fig, ax = plt.subplots(figsize=(11.0, 7.0))
    for rho in (0.3, 0.6):
        for beta_phi, ls in BETA_PHI_LINESTYLES.items():
            eps = sorted(e for (b, e, r, c) in recs if b == beta_phi and r == rho and c == "vary")
            if not eps:
                continue
            ax.plot(eps, [recs[(beta_phi, e, rho, "vary")]["R1"] for e in eps], color=RHO_COLORS[rho], linestyle=ls,
                    marker=BETA_PHI_MARKERS[beta_phi], markersize=8, linewidth=2.0)
        eps = sorted(e for (b, e, r, c) in recs if b == 0.01 and r == rho and c == "max")
        ax.plot(eps, [recs[(0.01, e, rho, "max")]["R1"] for e in eps], color=RHO_COLORS[rho], linewidth=1.3, alpha=0.45,
                marker="none", linestyle="-")
    ax.set_xlabel(r"$\beta_s$ in mm")
    ax.set_ylabel(r"$R_y^{(1)}$ in N/mm")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(0.012, 0.063)
    style(ax)
    beta_handles = [Line2D([0], [0], color="#2f2f2f", linestyle=ls, marker=BETA_PHI_MARKERS[b], markersize=8, linewidth=2.0,
                           label=rf"{CASE_LABELS['vary']}, $\beta_\phi={b:g}\,\mathrm{{mm^2}}$") for b, ls in BETA_PHI_LINESTYLES.items()]
    beta_handles.append(Line2D([0], [0], color="#2f2f2f", linewidth=1.3, alpha=0.45, linestyle="-",
                               label=rf"{CASE_LABELS['max']} (reference)"))
    rho_handles = [Line2D([0], [0], color=RHO_COLORS[rho], linewidth=2.4, label=rf"$\rho={rho:g}$") for rho in (0.3, 0.6)]
    first = ax.legend(handles=beta_handles, fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=rho_handles, fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save(fig, output_dir, "Beta_comparison_first_peak_vs_beta_s_M1e5")


def fig_mobility_convergence(mob, output_dir):
    """mob: {(epsilon, rho, case): {M: (R_traction_first_peak, Ds_over_W, u1)}}"""
    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(17.0, 6.8))
    markers = {0.03: "o", 0.06: "D"}
    fits = []
    for (epsilon, rho, case), series in sorted(mob.items()):
        if CAMPAIGN_MOBILITY not in series or len(series) < 2:
            continue
        c = COLORS[(rho, case)]
        ms = sorted(series)
        ref = series[CAMPAIGN_MOBILITY][0]
        values = np.array([series[m][0] for m in ms]) / ref
        ax_r.plot(ms, values, color=c, marker=markers.get(epsilon, "o"), markersize=10, linewidth=1.6,
                  linestyle="-" if epsilon == 0.03 else (0, (6.0, 3.0)), alpha=1.0 if epsilon == 0.03 else 0.8)
        ax_d.plot(ms, [series[m][1] for m in ms], color=c, marker=markers.get(epsilon, "o"), markersize=10, linewidth=1.6,
                  linestyle="-" if epsilon == 0.03 else (0, (6.0, 3.0)), alpha=1.0 if epsilon == 0.03 else 0.8)
        if len(ms) >= 3:
            # the power law holds for large M: fit on M >= 1e3 when three such points exist, else on all points
            fit_ms = [m for m in ms if m >= 1000.0]
            if len(fit_ms) < 3:
                fit_ms = ms
            x = np.array(fit_ms, dtype=float) ** (-2.0 / 3.0)
            A = np.vstack([np.ones_like(x), x]).T
            (r_inf, coef), *_ = np.linalg.lstsq(A, np.array([series[m][0] for m in fit_ms]), rcond=None)
            ms = fit_ms
            mm = np.logspace(np.log10(min(ms)), np.log10(3.0e5), 200)
            ax_r.plot(mm, (r_inf + coef * mm ** (-2.0 / 3.0)) / ref, color=c, linewidth=1.5, linestyle=(0, (2.0, 2.0)), alpha=0.7)
            fits.append({"beta_s": epsilon, "rho": rho, "case": CASE_TEXT[case], "R_inf_traction": r_inf, "c": coef,
                         "R_1e5_over_R_inf": ref / r_inf, "n_points": len(ms), "M_values": " ".join(f"{m:g}" for m in ms)})
    ax_r.set_xscale("log")
    ax_r.set_xlabel(r"$M$ in mm$^3$/(Nmm\,s)")
    ax_r.set_ylabel(r"$R_y^{(1)}(M)\,/\,R_y^{(1)}(M=10^5)$")
    ax_r.axhline(1.0, color="#7f7f7f", linewidth=0.8)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(r"$M$ in mm$^3$/(Nmm\,s)")
    ax_d.set_ylabel(r"$D_s/W$ at the first peak")
    for ax, label in ((ax_r, "a"), (ax_d, "b")):
        style(ax)
        panel_label(ax, label)
    ax_r.set_ylim(0.9, 1.1 * ax_r.get_ylim()[1])
    ax_r.set_xlim(50.0, 3.0e5)
    ax_d.set_xlim(50.0, 3.0e5)
    lo, hi = ax_d.get_ylim()
    ax_d.set_ylim(lo, hi * 4.0)
    extra = [Line2D([0], [0], color="#2f2f2f", marker="o", markersize=9, linewidth=1.6, label=r"$\beta_s=0.03$ mm"),
             Line2D([0], [0], color="#2f2f2f", marker="D", markersize=9, linewidth=1.6, linestyle=(0, (6.0, 3.0)), alpha=0.8,
                    label=r"$\beta_s=0.06$ mm"),
             Line2D([0], [0], color="#2f2f2f", linewidth=1.5, linestyle=(0, (2.0, 2.0)), alpha=0.7, label=r"fit $R_\infty + c\,M^{-2/3}$")]
    fig.legend(handles=structure_handles_energy() + extra, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               fontsize=LEGEND_FONT_SIZE, frameon=True)
    fig.tight_layout(w_pad=3.0)
    save(fig, output_dir, "Mobility_convergence_first_peak")
    return fits


# ----------------------------------------------------------------------------------------------- tables

def structure_name(rho, case):
    return rf"$\rho={rho:g}$, {CASE_LABELS[case]}"


def write_table(path, header, rows, caption_note=""):
    lines = [f"% {caption_note}".rstrip(), r"\begin{tabular}{" + "l" * 1 + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        if row is None:
            lines.append(r"\addlinespace")
        else:
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")
    print(f"[INFO] wrote {Path(path).name}")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def pct(a, b):
    return 100.0 * (a / b - 1.0)


def fmt_pct(v):
    return rf"${v:+.0f}\,\%$"


def tables(recs, output_dir, beta_phi=0.01):
    eps_values = sorted({e for (b, e, r, c) in recs if b == beta_phi})
    # ---- first-peak measures
    rows, csv_rows = [], []
    for e in eps_values:
        for rho, case in STRUCTURES:
            r = recs.get((beta_phi, e, rho, case))
            if r is None:
                continue
            rows.append([f"{e:g}", structure_name(rho, case), f"{r['R1']:.1f}", f"{r['u1']:.4f}", f"{r['W1']:.3f}",
                         f"{r['Pi_el1']:.3f}", f"{r['Pi_frac1']:.4f}"])
            csv_rows.append({"beta_phi": beta_phi, "beta_s": e, "rho": rho, "case": CASE_TEXT[case], **{k: r[k] for k in
                             ("R1", "u1", "W1", "Pi_el1", "Pi_frac1", "D_s1", "balance1", "Rmax", "u_Rmax", "Rmax_over_R1",
                              "ended_rising", "u_end", "W_end", "Pi_el_end", "Pi_frac_end", "D_s_end", "balance_end",
                              "Ds_over_W1", "Ds_share1", "Ds_over_W_end", "Ds_share_end")}})
        rows.append(None)
    write_table(output_dir / "table_first_peak_M1e5.tex",
                [r"$\beta_s$ [mm]", "structure", r"$R_y^{(1)}$ [N/mm]", r"$u_y^{(1)}$ [mm]", r"$W^{(1)}$ [Nmm/mm]",
                 r"$\Pi_\mathrm{el}^{(1)}$ [Nmm/mm]", r"$\Pi_\mathrm{frac}^{(1)}$ [Nmm/mm]"], rows[:-1],
                "first-peak measures, M = 1e5, beta_phi = 0.01, total of both strips; source 15_manuscript_figures_M1e5.py")
    write_csv(output_dir / "manuscript_runs_M1e5.csv", csv_rows)

    # ---- secondary measures
    rows = []
    for e in eps_values:
        for rho, case in STRUCTURES:
            r = recs.get((beta_phi, e, rho, case))
            if r is None:
                continue
            rising = r"$^{\uparrow}$" if r["ended_rising"] else ""
            rows.append([f"{e:g}", structure_name(rho, case), f"{r['Rmax']:.1f}{rising}", f"{r['Rmax_over_R1']:.2f}",
                         f"{r['u_Rmax']:.4f}", f"{r['W_end']:.3f}", f"{r['Pi_frac_end'] + r['D_s_end']:.3f}", f"{r['u_end']:.4f}"])
        rows.append(None)
    write_table(output_dir / "table_secondary_M1e5.tex",
                [r"$\beta_s$ [mm]", "structure", r"$\max R_y$ [N/mm]", r"$\max R_y / R_y^{(1)}$", r"$u_y(\max R_y)$ [mm]",
                 r"$W_\mathrm{end}$ [Nmm/mm]", r"$(\Pi_\mathrm{frac}+D_s)_\mathrm{end}$ [Nmm/mm]", r"$u_{y,\mathrm{end}}$ [mm]"], rows[:-1],
                "secondary measures up to u_y = 0.03 mm (E_min beta_s = 0.015: 0.0284 mm); arrow = still rising at the end")

    # ---- dissipation ratios (all beta_s)
    rows = []
    for e in eps_values:
        for rho, case in STRUCTURES:
            r = recs.get((beta_phi, e, rho, case))
            if r is None:
                continue
            rows.append([f"{e:g}", structure_name(rho, case), f"{100 * r['Ds_over_W1']:.3f}", f"{100 * r['Ds_share1']:.1f}",
                         f"{100 * r['Ds_over_W_end']:.1f}", f"{100 * r['Ds_share_end']:.1f}", f"{100 * (r['balance1'] - 1):+.2f}",
                         f"{100 * (r['balance_end'] - 1):+.2f}"])
        rows.append(None)
    write_table(output_dir / "table_dissipation_M1e5.tex",
                [r"$\beta_s$ [mm]", "structure", r"$D_s/W$ at $R_y^{(1)}$ [\%]", r"$D_s/(\Pi_\mathrm{frac}+D_s)$ at $R_y^{(1)}$ [\%]",
                 r"$D_s/W$ at end [\%]", r"$D_s/(\Pi_\mathrm{frac}+D_s)$ at end [\%]", r"$W/\Pi_\mathrm{tot}-1$ at $R_y^{(1)}$ [\%]",
                 r"$W/\Pi_\mathrm{tot}-1$ at end [\%]"], rows[:-1],
                "mobility dissipation and energy balance, M = 1e5, beta_phi = 0.01")

    # ---- comparisons
    comp_rows, comp_csv = [], []
    for rho, case, ref in COMPARISONS:
        for key, label in (("R1", "first-peak load"), ("W1", "work at first peak"), ("Rmax", r"maximum load to $u_y=0.03$ mm"),
                           ("W_end", r"work to $u_y=0.03$ mm")):
            vals = []
            for e in eps_values:
                a, b = recs.get((beta_phi, e, rho, case)), recs.get((beta_phi, e, rho, ref))
                vals.append(pct(a[key], b[key]) if a and b else np.nan)
            comp_rows.append([rf"$\rho={rho:g}$: {CASE_LABELS[case]} vs {CASE_LABELS[ref]}", label] + [fmt_pct(v) for v in vals])
            comp_csv.append({"rho": rho, "comparison": f"{CASE_TEXT[case]} vs {CASE_TEXT[ref]}", "measure": key,
                             **{f"beta_s_{e:g}": v for e, v in zip(eps_values, vals)}, "min": np.nanmin(vals), "max": np.nanmax(vals)})
        comp_rows.append(None)
    write_table(output_dir / "table_comparisons_M1e5.tex", ["comparison", "measure"] + [rf"$\beta_s={e:g}$" for e in eps_values],
                comp_rows[:-1], "relative differences, M = 1e5, beta_phi = 0.01")
    write_csv(output_dir / "manuscript_comparisons_M1e5.csv", comp_csv)

    # ---- beta_phi study (E_var first peak, relative to E_max of the same rho and beta_s)
    rows, bcsv = [], []
    for rho in (0.3, 0.6):
        for e in eps_values:
            row = [f"{rho:g}", f"{e:g}"]
            ref = recs.get((0.01, e, rho, "max"))
            for b in (0.001, 0.01, 0.05):
                r = recs.get((b, e, rho, "vary"))
                if r is None or ref is None:
                    row.append("--")
                    continue
                row.append(rf"{r['R1']:.1f} ({pct(r['R1'], ref['R1']):+.0f}\,\%)")
                bcsv.append({"rho": rho, "beta_s": e, "beta_phi": b, "R1": r["R1"], "u1": r["u1"], "W1": r["W1"],
                             "R1_vs_Emax_pct": pct(r["R1"], ref["R1"]), "Rmax": r["Rmax"], "Rmax_vs_Emax_pct": pct(r["Rmax"], ref["Rmax"]),
                             "W1_vs_Emax_pct": pct(r["W1"], ref["W1"])})
            rows.append(row)
        rows.append(None)
    write_table(output_dir / "table_beta_phi_M1e5.tex",
                [r"$\rho$", r"$\beta_s$ [mm]", r"$\beta_\phi=0.001$", r"$\beta_\phi=0.01$", r"$\beta_\phi=0.05$"], rows[:-1],
                "E_var first-peak load in N/mm (relative to E_max of the same rho and beta_s), M = 1e5")
    write_csv(output_dir / "manuscript_beta_phi_M1e5.csv", bcsv)
    return csv_rows


# ----------------------------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign-root", type=Path, default=SCRIPT_DIR / "results" / "campaign_260930_M1e5")
    parser.add_argument("--reference-root", type=Path, default=SCRIPT_DIR / "results" / "new_W_whole_boundary",
                        help="M = 100 histories (submitted campaign) for the mobility-convergence figure")
    parser.add_argument("--mobility-root", type=Path, action="append", default=[],
                        help="roots with M = 1e3/1e4/1e5 histories (mobility_study_260929, mobility_convergence_260930)")
    parser.add_argument("--summary-csv", type=Path, default=SCRIPT_DIR / "68c3b8d0b7dca7b64b8b7a93" / "Images_A02" / "summary_260504.csv")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "plots" / "energy_consistent" / "manuscript")
    parser.add_argument("--no-usetex", action="store_true")
    args = parser.parse_args()
    plt.rcParams.update({"text.usetex": not args.no_usetex, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
                         "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": 15})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # campaign records (residual columns)
    recs = {}
    for (m, beta, eps, rho, case), path in ec.discover(args.campaign_root, 6, "spectral").items():
        if m != CAMPAIGN_MOBILITY:
            continue
        recs[(beta, eps, rho, case)] = campaign_record(path)
    print(f"[INFO] {len(recs)} campaign runs at M = {CAMPAIGN_MOBILITY:g}")

    fig_response_energy_grid(recs, 0.01, 0.03, args.output_dir)
    fig_peak_metrics_grid(recs, 0.01, args.output_dir)
    if args.summary_csv.exists():
        fig_first_peak_vs_sigma_c(recs, load_sigma_c(args.summary_csv), 0.01, args.output_dir)
    else:
        print(f"[WARNING] {args.summary_csv} not found, sigma_c figure skipped")
    fig_beta_comparison_curves(recs, 0.03, args.output_dir)
    fig_beta_comparison_first_peak(recs, args.output_dir)
    tables(recs, args.output_dir)

    # mobility convergence: traction-based first peak for every M (the traction/residual ratio is M-independent)
    mob = {}
    roots = [args.reference_root, args.campaign_root, *args.mobility_root]
    for root in roots:
        if not root.exists():
            print(f"[WARNING] {root} missing")
            continue
        for (m, beta, eps, rho, case), path in ec.discover(root, 6, "spectral").items():
            if beta != 0.01 or eps not in (0.03, 0.06):
                continue
            if eps == 0.06 and (rho, case) != (0.3, "min"):
                continue
            mob.setdefault((eps, rho, case), {})[m] = traction_first_peak(path)
    fits = fig_mobility_convergence(mob, args.output_dir)
    mob_rows = [{"beta_s": eps, "rho": rho, "case": CASE_TEXT[case], "M": m, "R1_traction": v[0], "Ds_over_W1": v[1], "u1": v[2],
                 "R1_over_R1_1e5": v[0] / series[CAMPAIGN_MOBILITY][0] if CAMPAIGN_MOBILITY in series else np.nan}
                for (eps, rho, case), series in sorted(mob.items()) for m, v in sorted(series.items())]
    write_csv(args.output_dir / "mobility_convergence_M1e5.csv", mob_rows)
    write_csv(args.output_dir / "mobility_convergence_fits.csv", fits)
    for f in fits:
        print(f"[FIT] beta_s={f['beta_s']:g} rho={f['rho']:g} {f['case']}: R_inf={f['R_inf_traction']:.2f}, "
              f"R(1e5)/R_inf={f['R_1e5_over_R_inf']:.4f}, M={f['M_values']}")


if __name__ == "__main__":
    main()
