#!/usr/bin/env python3
"""Evaluate the phase-field mobility side study (campaign 260929_mobility).

Compares the result_graphs_*.txt histories of the reference campaign
(M = 100, no _M suffix) with the new runs (suffix _M<label>) for the same
leaf datasets, split and beta_s. Needs numpy + matplotlib only (no h5py).

Outputs (default plots/mobility_study/):
  mobility_<dataset>_<case>.pdf/.png  four panels per case (overview)
  Ry_vs_uy_mobility_<dataset>.pdf/.png, Ry_vs_uy_mobility_all.pdf/.png
                                      force-displacement, 09_evaluation style
  Dissipation_grid_vs_uy_mobility_<dataset>.pdf/.png
                                      D_s, Pi_frac, dissipation rate, D_s/W (09 style)
  Dissipation_vs_uy_mobility_all.pdf/.png  D_s of all cases
  mobility_study_summary.csv / .md    peak and final-state metrics per (case, M)
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE_ROOT = Path(
    "/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/results/new_W_whole_boundary/"
    "simulation_20260526_122937_SPLITspectral_EPS0_03"
)
DEFAULT_STUDY_ROOT = Path(
    "/Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/mobility_study_260929"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots" / "mobility_study"
REFERENCE_MOBILITY = 100.0

# result_graphs_*.txt columns (0-based)
COL_T = 0
COL_UY = 1
COL_RY = 2
COL_D_RATE = 8  # integral of s_dot^2/M over the domain (rate of D_s)
COL_W_STRIPS = 4  # work on the two load strips; "max_Work" in summary_260504.csv
COL_PI_FRAC = 5
COL_PI_EL = 6
COL_D_S = 9
COL_W_TOTAL = 13  # total-boundary work, the W used in the manuscript

RESULT_RE = re.compile(
    r"^result_graphs_(?P<dataset>beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)_(?:var|min|max))"
    r"_(?P<case>vary|min|max)_(?P<split>spectral|volumetric)"
    r"_eps(?P<epsilon>[0-9_]+)(?:_M(?P<mobility>[0-9_]+))?\.txt$"
)

# Same colors as 09_evaluation_260504_parameter_space.py (ENERGY_DATASET_COLORS)
ENERGY_DATASET_COLORS = {
    (0.3, "min"): "#b2182b",
    (0.3, "max"): "#ef8a00",
    (0.3, "vary"): "#d6604d",
    (0.6, "min"): "#2166ac",
    (0.6, "max"): "#67a9cf",
    (0.6, "vary"): "#053061",
}
CASE_LABELS = {
    "min": r"$\mathbf{E}_{\mathrm{min}}$",
    "max": r"$\mathbf{E}_{\mathrm{max}}$",
    "vary": r"$\mathbf{E}_{\mathrm{var}}$",
}
QUANTITY_COLORS = {"el": "#2f6fb7", "frac": "#c17c2a", "ds": "#7b5ab6"}
MOBILITY_LINESTYLES = {
    100.0: "-",
    300.0: (0, (1.2, 2.2)),
    1000.0: (0, (7.0, 3.0)),
    3000.0: (0, (10.0, 2.0, 2.0, 2.0, 2.0, 2.0)),
    10000.0: (0, (4.0, 2.0, 1.4, 2.0)),
}
LOADING_VELOCITY = 1.0  # mm/s, u_y = -v0 t
# 09_evaluation_260504_parameter_space.py figure style
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 22
LEGEND_FONT_SIZE = 22
TITLE_FONT_SIZE = 24
STYLE_LINEWIDTH = 1.8
STYLE_MARKER_SIZE = 95
RATIO_DENOMINATOR_MIN = 1.0e-4  # Nmm/mm, plotting only
PEAK_MARKER_SIZE = 70
PEAK_MARKER_EDGE_COLOR = "#1f1f1f"


def label_to_float(label):
    return float(label.replace("m", "-").replace("_", "."))


def load_history(path):
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[1] <= COL_W_TOTAL:
        raise ValueError(f"{path}: expected at least {COL_W_TOTAL + 1} columns, got {data.shape[1]}")
    return data


def discover(root, epsilon, split, mobility_required):
    """Return {(dataset, case, mobility): path} below root."""
    found = {}
    for path in sorted(root.rglob("result_graphs_*.txt")):
        match = RESULT_RE.match(path.name)
        if not match or match["split"] != split:
            continue
        if not math.isclose(label_to_float(match["epsilon"]), epsilon):
            continue
        has_mobility = match["mobility"] is not None
        if has_mobility != mobility_required:
            continue
        mobility = label_to_float(match["mobility"]) if has_mobility else REFERENCE_MOBILITY
        key = (match["dataset"], match["case"], mobility)
        if key in found:
            raise RuntimeError(f"Duplicate history for {key}: {found[key]} and {path}")
        found[key] = path
    return found


def safe_ratio(numerator, denominator):
    return numerator / denominator if abs(denominator) > 0.0 else float("nan")


def metrics(data):
    uy = np.abs(data[:, COL_UY])
    ry = np.abs(data[:, COL_RY])
    work = np.abs(data[:, COL_W_TOTAL])
    frac = data[:, COL_PI_FRAC]
    d_s = data[:, COL_D_S]
    peak = int(np.argmax(ry))
    return {
        "max_Ry": ry[peak],
        "uy_peak": uy[peak],
        "W_peak": work[peak],
        "W_strips_peak": abs(data[peak, COL_W_STRIPS]),
        "Pi_frac_peak": frac[peak],
        "D_s_peak": d_s[peak],
        "D_s_over_W_peak": safe_ratio(d_s[peak], work[peak]),
        "D_s_over_frac_plus_D_s_peak": safe_ratio(d_s[peak], frac[peak] + d_s[peak]),
        "D_s_over_W_final": safe_ratio(d_s[-1], work[-1]),
        "D_s_over_frac_plus_D_s_final": safe_ratio(d_s[-1], frac[-1] + d_s[-1]),
        "n_steps": data.shape[0],
        "uy_final": uy[-1],
    }


def rho_from_dataset(dataset):
    return label_to_float(re.search(r"_rho_(0_\d+)_", dataset + "_").group(1))


def mobility_text(mobility):
    return f"{mobility:g}"


def mobility_linestyle(mobility):
    for known, style in MOBILITY_LINESTYLES.items():
        if math.isclose(mobility, known):
            return style
    return (0, (2.0, 1.0))


def case_color(dataset, case):
    return ENERGY_DATASET_COLORS.get((rho_from_dataset(dataset), case), "#5e6670")


def case_legend_label(dataset, case):
    return rf"$\rho={rho_from_dataset(dataset):g}$, {CASE_LABELS.get(case, case)}"


def style_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3)


def add_time_axis(ax):
    """Secondary x axis t = |u_y| / v0 (identical numbers for v0 = 1 mm/s)."""
    top = ax.secondary_xaxis(
        "top", functions=(lambda u: u / LOADING_VELOCITY, lambda t: t * LOADING_VELOCITY)
    )
    top.set_xlabel(r"$t$ in s", fontsize=AXIS_LABEL_SIZE)
    top.tick_params(labelsize=TICK_LABEL_SIZE)


def mark_peak(ax, uy, values, peak, color):
    ax.scatter(
        uy[peak], values[peak], s=STYLE_MARKER_SIZE, marker="X", color=color,
        edgecolor=PEAK_MARKER_EDGE_COLOR, linewidth=0.85, zorder=5,
    )


def mobility_handles(mobilities, include_peak=True):
    handles = [
        Line2D([0], [0], color="#2f2f2f", linestyle=mobility_linestyle(m), linewidth=2.2,
               label=rf"$M={mobility_text(m)}$")
        for m in mobilities
    ]
    if include_peak:
        handles.append(Line2D([0], [0], marker="X", color="#2f2f2f", linewidth=0.0, markersize=9,
                              label=r"$R_y=\max R_y$"))
    return handles


def case_handles(cases):
    return [
        Line2D([0], [0], color=case_color(d, c), linewidth=2.4, label=case_legend_label(d, c))
        for d, c in cases
    ]


def save_figure(fig, output_dir, stem):
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return stem


def plot_curves_vs_uy(cases, column, ylabel, output_dir, stem, headroom=1.05):
    """One quantity vs u_y; color = (rho, case) as in 09_evaluation, line style = M."""
    fig, ax = plt.subplots(figsize=(10.0, 6.2))
    max_uy = 0.0
    max_value = 0.0
    mobilities = set()
    for (dataset, case), runs in cases.items():
        color = case_color(dataset, case)
        for mobility, data in runs:
            uy = np.abs(data[:, COL_UY])
            values = np.abs(data[:, column])
            max_value = max(max_value, float(np.max(values)))
            peak = int(np.argmax(np.abs(data[:, COL_RY])))
            ax.plot(uy, values, color=color, linestyle=mobility_linestyle(mobility), linewidth=STYLE_LINEWIDTH)
            mark_peak(ax, uy, values, peak, color)
            max_uy = max(max_uy, float(uy[-1]))
            mobilities.add(mobility)
    ax.set_xlim(0.0, max_uy * 1.02)
    ax.set_ylim(0.0, max_value * headroom)  # headroom keeps the legends clear of the curves
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(ylabel)
    style_axes(ax)
    add_time_axis(ax)
    legend_m = ax.legend(handles=mobility_handles(sorted(mobilities)), loc="upper right",
                         fontsize=LEGEND_FONT_SIZE - 4)
    if len(cases) > 1:
        ax.add_artist(legend_m)
        ax.legend(handles=case_handles(cases), loc="upper left", fontsize=LEGEND_FONT_SIZE - 4)
    else:
        (dataset, case), = cases
        ax.set_title(case_legend_label(dataset, case) + r"; $M$ in mm$^3$/(Nmm\,s)",
                     fontsize=TITLE_FONT_SIZE - 2, pad=14)
    return save_figure(fig, output_dir, stem)


def plot_dissipation_grid(dataset, case, runs, output_dir):
    """(a) D_s, (b) Pi_frac (same scale), (c) dissipation rate (log), (d) D_s/W; all vs u_y (= t)."""
    color = case_color(dataset, case)
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.5), sharex=True)
    ax_ds, ax_frac, ax_rate, ax_ratio = axes.ravel()
    max_uy = 0.0
    top = 0.0
    for mobility, data in runs:
        linestyle = mobility_linestyle(mobility)
        uy = np.abs(data[:, COL_UY])
        d_s = data[:, COL_D_S]
        frac = data[:, COL_PI_FRAC]
        rate = np.where(data[:, COL_D_RATE] > 0.0, data[:, COL_D_RATE], np.nan)
        work = np.abs(data[:, COL_W_TOTAL])
        with np.errstate(divide="ignore", invalid="ignore"):
            ds_over_w = np.where(work > RATIO_DENOMINATOR_MIN, d_s / work, np.nan)
        peak = int(np.argmax(np.abs(data[:, COL_RY])))
        for ax, values in ((ax_ds, d_s), (ax_frac, frac), (ax_rate, rate), (ax_ratio, ds_over_w)):
            ax.plot(uy, values, color=color, linestyle=linestyle, linewidth=STYLE_LINEWIDTH)
            if np.isfinite(values[peak]):
                mark_peak(ax, uy, values, peak, color)
        max_uy = max(max_uy, float(uy[-1]))
        top = max(top, float(np.max(d_s)), float(np.max(frac)))

    ax_ds.set_ylabel(r"$D_s$ in Nmm/mm")
    ax_frac.set_ylabel(r"$\Pi_\mathrm{frac}$ in Nmm/mm")
    ax_rate.set_ylabel(r"$\dot D_s$ in Nmm/(mm\,s)")
    ax_rate.set_yscale("log")
    ax_ratio.set_ylabel(r"$D_s/W$")
    ax_ratio.set_ylim(0.0, 1.0)
    for ax in (ax_ds, ax_frac):  # same scale for D_s and Pi_frac
        ax.set_ylim(0.0, top * 1.05)
    ax_rate.set_ylim(bottom=1.0e-4)  # first load steps (~1e-20) are not of interest
    for ax in (ax_rate, ax_ratio):
        ax.set_xlabel(r"$u_y$ in mm")
    for panel_label, ax in zip("abcd", axes.ravel()):
        ax.text(0.02, 0.95, rf"\textbf{{({panel_label})}}", transform=ax.transAxes,
                ha="left", va="top", fontsize=TITLE_FONT_SIZE,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0})
        ax.set_xlim(0.0, max_uy * 1.02)
        style_axes(ax)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.legend(handles=mobility_handles([m for m, _ in runs]), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=len(runs) + 1, fontsize=LEGEND_FONT_SIZE,
               title=case_legend_label(dataset, case) + r"; $M$ in mm$^3$/(Nmm\,s); "
               r"$t = u_y/v_0$ with $v_0 = 1$ mm/s, i.e. $u_y$ in mm $=$ $t$ in s",
               title_fontsize=LEGEND_FONT_SIZE - 2)
    return save_figure(fig, output_dir, f"Dissipation_grid_vs_uy_mobility_{dataset}")


def plot_case(dataset, case, runs, output_dir, dpi):
    """runs: list of (mobility, data) sorted by mobility."""
    rho = rho_from_dataset(dataset)
    case_color = ENERGY_DATASET_COLORS.get((rho, case), "#5e6670")
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5), constrained_layout=True)
    ax_ry, ax_w, ax_en, ax_ratio = axes.ravel()

    for index, (mobility, data) in enumerate(runs):
        linestyle = mobility_linestyle(mobility)
        uy = np.abs(data[:, COL_UY])
        ry = np.abs(data[:, COL_RY])
        work = np.abs(data[:, COL_W_TOTAL])
        frac = data[:, COL_PI_FRAC]
        elastic = data[:, COL_PI_EL]
        d_s = data[:, COL_D_S]
        peak = int(np.argmax(ry))
        # Ratios of ~1e-20 energies before damage onset are meaningless -> masked.
        with np.errstate(divide="ignore", invalid="ignore"):
            ds_over_w = np.where(work > RATIO_DENOMINATOR_MIN, d_s / work, np.nan)
            ds_share = np.where(frac + d_s > RATIO_DENOMINATOR_MIN, d_s / (frac + d_s), np.nan)

        series = [
            (ax_ry, [(ry, case_color)]),
            (ax_w, [(work, case_color)]),
            (ax_en, [(elastic, QUANTITY_COLORS["el"]), (frac, QUANTITY_COLORS["frac"]), (d_s, QUANTITY_COLORS["ds"])]),
            (ax_ratio, [(ds_over_w, QUANTITY_COLORS["el"]), (ds_share, QUANTITY_COLORS["frac"])]),
        ]
        for ax, curves in series:
            for values, color in curves:
                ax.plot(uy, values, color=color, linestyle=linestyle, linewidth=2.0)
                ax.scatter(
                    uy[peak], values[peak], marker="X", s=PEAK_MARKER_SIZE, color=color,
                    edgecolor=PEAK_MARKER_EDGE_COLOR, linewidth=0.8, zorder=5,
                )

    ax_ry.set_ylabel(r"$R_y$ in N/mm")
    ax_w.set_ylabel(r"$W$ in Nmm/mm")
    ax_en.set_ylabel(r"energy in Nmm/mm")
    ax_ratio.set_ylabel(r"ratio")
    ax_ratio.set_ylim(0.0, 1.0)
    for ax, tag in zip(axes.ravel(), "abcd"):
        ax.set_xlabel(r"$|u_y|$ in mm")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(True, color="#d9d9d9", linewidth=0.6)
        ax.set_title(f"({tag})", loc="left")

    mobility_handles = [
        Line2D([], [], color="#3a3a3a", linestyle=mobility_linestyle(m), linewidth=2.0,
               label=rf"$M = {mobility_text(m)}$")
        for m, _ in runs
    ]
    peak_handle = Line2D([], [], color="#3a3a3a", marker="X", linestyle="none", markersize=9,
                         markeredgecolor=PEAK_MARKER_EDGE_COLOR, label=r"peak $\max R_y$")
    ax_ry.legend(handles=mobility_handles + [peak_handle], loc="best")
    ax_en.legend(
        handles=[
            Line2D([], [], color=QUANTITY_COLORS["el"], linewidth=2.0, label=r"$\Pi_\mathrm{el}$"),
            Line2D([], [], color=QUANTITY_COLORS["frac"], linewidth=2.0, label=r"$\Pi_\mathrm{frac}$"),
            Line2D([], [], color=QUANTITY_COLORS["ds"], linewidth=2.0, label=r"$D_s$"),
        ],
        loc="upper left",
    )
    ax_ratio.legend(
        handles=[
            Line2D([], [], color=QUANTITY_COLORS["el"], linewidth=2.0, label=r"$D_s/W$"),
            Line2D([], [], color=QUANTITY_COLORS["frac"], linewidth=2.0, label=r"$D_s/(\Pi_\mathrm{frac}+D_s)$"),
        ],
        loc="upper left",
    )
    fig.suptitle(
        rf"{CASE_LABELS.get(case, case)}, $\rho = {rho:g}$; $M$ in mm$^3$/(Nmm\,s)"
    )
    stem = f"mobility_{dataset}_{case}"
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi)
    plt.close(fig)
    return stem


SUMMARY_FIELDS = [
    ("dataset", "dataset", None),
    ("case", "case", None),
    ("M", "M", "{:g}"),
    ("max_Ry", r"max R_y [N/mm]", "{:.3f}"),
    ("uy_peak", "u_y at peak [mm]", "{:.6f}"),
    ("W_peak", "W at peak [Nmm/mm]", "{:.4f}"),
    ("W_strips_peak", "W_strips at peak", "{:.4f}"),
    ("Pi_frac_peak", "Pi_frac at peak", "{:.4f}"),
    ("D_s_peak", "D_s at peak", "{:.4f}"),
    ("D_s_over_W_peak", "D_s/W peak", "{:.4f}"),
    ("D_s_over_frac_plus_D_s_peak", "D_s/(Pi_frac+D_s) peak", "{:.3f}"),
    ("D_s_over_W_final", "D_s/W final", "{:.3f}"),
    ("D_s_over_frac_plus_D_s_final", "D_s/(Pi_frac+D_s) final", "{:.3f}"),
    ("n_steps", "steps", "{:d}"),
    ("uy_final", "final u_y [mm]", "{:.6f}"),
]


def write_summary(rows, output_dir):
    csv_path = output_dir / "mobility_study_summary.csv"
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([key for key, _, _ in SUMMARY_FIELDS])
        for row in rows:
            writer.writerow([row[key] for key, _, _ in SUMMARY_FIELDS])

    md_path = output_dir / "mobility_study_summary.md"
    lines = [
        "| " + " | ".join(header for _, header, _ in SUMMARY_FIELDS) + " |",
        "|" + "|".join("---" for _ in SUMMARY_FIELDS) + "|",
    ]
    for row in rows:
        cells = [fmt.format(row[key]) if fmt else str(row[key]) for key, _, fmt in SUMMARY_FIELDS]
        lines.append("| " + " | ".join(cells) + " |")
    md_path.write_text("\n".join(lines) + "\n")
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT,
                        help="M = 100 campaign folder (files without _M suffix)")
    parser.add_argument("--study-root", type=Path, default=DEFAULT_STUDY_ROOT,
                        help="root of the 260929_mobility campaign folders (files with _M suffix)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epsilon", type=float, default=0.03, help="beta_s in mm")
    parser.add_argument("--split", default="spectral")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-usetex", action="store_true", help="use mathtext instead of LaTeX")
    args = parser.parse_args()

    plt.rcParams.update({
        "text.usetex": not args.no_usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "font.size": 15,
        "legend.fontsize": 13,
    })

    study = discover(args.study_root, args.epsilon, args.split, mobility_required=True)
    if not study:
        raise SystemExit(f"No _M result histories for eps={args.epsilon} below {args.study_root}")
    reference = discover(args.reference_root, args.epsilon, args.split, mobility_required=False)
    studied_cases = sorted({(dataset, case) for dataset, case, _ in study})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cases = {}
    for dataset, case in studied_cases:
        paths = {m: p for (d, c, m), p in {**reference, **study}.items() if (d, c) == (dataset, case)}
        if REFERENCE_MOBILITY not in paths:
            print(f"[WARNING] No M = {REFERENCE_MOBILITY:g} reference for {dataset} {case}")
        runs = []
        for mobility in sorted(paths):
            data = load_history(paths[mobility])
            runs.append((mobility, data))
            rows.append({"dataset": dataset, "case": case, "M": mobility, **metrics(data)})
            print(f"[INFO] {dataset} {case} M={mobility:g}: {paths[mobility]}")
        cases[(dataset, case)] = runs
        stems = [
            plot_case(dataset, case, runs, args.output_dir, args.dpi),
            plot_curves_vs_uy({(dataset, case): runs}, COL_RY, r"$R_y$ in N/mm",
                              args.output_dir, f"Ry_vs_uy_mobility_{dataset}"),
            plot_dissipation_grid(dataset, case, runs, args.output_dir),
        ]
        for stem in stems:
            print(f"[INFO] Wrote {args.output_dir / stem}.pdf/.png")

    for column, ylabel, stem, headroom in (
        (COL_RY, r"$R_y$ in N/mm", "Ry_vs_uy_mobility_all", 1.05),
        (COL_D_S, r"$D_s$ in Nmm/mm", "Dissipation_vs_uy_mobility_all", 1.5),
    ):
        plot_curves_vs_uy(cases, column, ylabel, args.output_dir, stem, headroom)
        print(f"[INFO] Wrote {args.output_dir / stem}.pdf/.png")

    csv_path, md_path = write_summary(rows, args.output_dir)
    print(f"[INFO] Wrote {csv_path}")
    print(f"[INFO] Wrote {md_path}")
    print(md_path.read_text())


if __name__ == "__main__":
    main()
