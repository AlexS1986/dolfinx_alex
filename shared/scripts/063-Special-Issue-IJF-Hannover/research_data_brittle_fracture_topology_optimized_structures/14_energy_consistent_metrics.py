#!/usr/bin/env python3
"""Energy-consistent peak metrics from existing result_graphs_*.txt histories (no rerun).

Background (MOBILITY_STUDY_FINDINGS.md, section 8): the reaction force in columns 2 and 7 is a traction
integral over the strip facets and underestimates the force by 24-38 %; the traction-based works in
columns 4 (two strips) and 13 (whole boundary) are inaccurate as well. The histories nevertheless contain
the exact external work through the energy balance,

    W_exact = Pi_el + Pi_frac + D_s          (columns 6 + 5 + 9),

and the energy-consistent total reaction of both strips follows from R = dW_exact/du_y. Up to the peak it is
evaluated here as R_i = R_traction,i / c_i with the calibration factor

    c_i = int_{u_i - w}^{u_i} R_traction du / (W_exact(u_i) - W_exact(u_i - w)),   w = WINDOW * u_peak,

i.e. the smooth traction curve gives the shape and the energy balance gives the level. Checked against a run
with residual-based reactions (columns 14/15): deviation of the peak load 0.3 %. After the peak the
reconstruction is not reliable (load drops), so curves are only given up to the peak.
Runs that already contain columns 14-17 are evaluated directly from them.

All reaction forces reported here are the TOTAL of both strips.

Peak definition (decision 2026-09-30): the peak is the FIRST peak of the total reaction, i.e. the first
running maximum after which the load falls by more than FIRST_PEAK_DROP (1 %) before exceeding it again
(first crack event). If the load later rises above the first peak (partial drop with crack arrest, e.g.
rho = 0.6 E_var at high mobility), the ratio "max/first peak" is reported as well; it is a lower bound of the
ultimate load when the run ended with the load still rising.

Usage:
  python3 14_energy_consistent_metrics.py [RESULT_ROOT] [--extra-root ROOT ...] [--output-dir DIR]
Outputs (default plots/energy_consistent/): energy_consistent_runs.csv, energy_consistent_comparisons.csv,
energy_consistent_summary.md and figures (pdf/png).
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.legend
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = SCRIPT_DIR / "results" / "new_W_whole_boundary"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "plots" / "energy_consistent"
REFERENCE_MOBILITY = 100.0
WINDOW = 0.1  # calibration window as a fraction of u_y at the peak
FIRST_PEAK_DROP = 0.01  # relative load drop that ends the search for the first peak

COL_UY, COL_RY_L, COL_W_STRIPS, COL_PI_FRAC, COL_PI_EL, COL_RY_R, COL_D_S, COL_W_BOUNDARY = 1, 2, 4, 5, 6, 7, 9, 13
COL_RES_L, COL_RES_R, COL_W_RES = 14, 15, 17

RESULT_RE = re.compile(
    r"^result_graphs_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)_(?:var|min|max)"
    r"_(?P<case>vary|min|max)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_]+?))?(?:_M(?P<mobility>[0-9_]+))?\.txt$"
)
EPS_DIR_RE = re.compile(r"_EPS(\d+_\d+)")

# colors and fonts as in 09_evaluation_260504_parameter_space.py / 12_mobility_study_eval.py
ENERGY_DATASET_COLORS = {
    (0.3, "min"): "#b2182b", (0.3, "max"): "#ef8a00", (0.3, "vary"): "#d6604d",
    (0.6, "min"): "#2166ac", (0.6, "max"): "#67a9cf", (0.6, "vary"): "#053061",
}
CASE_LABELS = {"min": r"$\mathbf{E}_{\mathrm{min}}$", "max": r"$\mathbf{E}_{\mathrm{max}}$",
               "vary": r"$\mathbf{E}_{\mathrm{var}}$"}
CASE_TEXT = {"min": "E_min", "max": "E_max", "vary": "E_var"}
AXIS_LABEL_SIZE, TICK_LABEL_SIZE, LEGEND_FONT_SIZE = 24, 22, 17
STRUCTURES = [(0.3, "max"), (0.3, "min"), (0.3, "vary"), (0.6, "max"), (0.6, "vary")]
COMPARISONS = [(0.3, "vary", "min"), (0.3, "vary", "max"), (0.6, "vary", "max")]

_trapz = getattr(np, "trapezoid", None) or np.trapz


def label_to_float(label):
    return float(label.replace("_", "."))


def discover(root, a_value, split):
    runs = {}
    for path in sorted(Path(root).rglob("result_graphs_*.txt")):
        match = RESULT_RE.match(path.name)
        if not match or int(match["a"]) != a_value or match["split"] != split:
            continue
        if match["epsilon"]:
            epsilon = label_to_float(match["epsilon"])
        else:
            eps_dirs = EPS_DIR_RE.findall(str(path))
            if not eps_dirs:
                print(f"[WARNING] no beta_s found for {path}")
                continue
            epsilon = label_to_float(eps_dirs[-1])
        mobility = label_to_float(match["mobility"]) if match["mobility"] else REFERENCE_MOBILITY
        key = (mobility, label_to_float(match["beta"]), epsilon, label_to_float(match["rho"]), match["case"])
        runs[key] = path
    return runs


def first_peak_index(reaction):
    """First running maximum followed by a drop of more than FIRST_PEAK_DROP; global maximum if there is none."""
    best = 0
    for i in range(1, len(reaction)):
        if reaction[i] > reaction[best]:
            best = i
        elif reaction[i] < (1.0 - FIRST_PEAK_DROP) * reaction[best]:
            return best
    return int(np.argmax(reaction))


def calibration_factor(u, r_traction, w_exact, i, window):
    j = int(np.searchsorted(u, u[i] - window, side="right")) - 1
    j = max(j, 0)
    if j >= i or w_exact[i] <= w_exact[j]:
        return np.nan
    return _trapz(r_traction[j:i + 1], u[j:i + 1]) / (w_exact[i] - w_exact[j])


def evaluate(path):
    data = np.loadtxt(path, comments="#")
    u = np.abs(data[:, COL_UY])
    r_left = np.abs(data[:, COL_RY_L])
    r_traction = np.abs(data[:, COL_RY_L] + data[:, COL_RY_R])
    w_exact = data[:, COL_PI_EL] + data[:, COL_PI_FRAC] + data[:, COL_D_S]
    has_residual = data.shape[1] > COL_W_RES

    peak_left = int(np.argmax(r_left))          # peak index used in the manuscript (left strip, global maximum)
    peak = first_peak_index(r_traction)          # first peak of the total traction-based reaction
    window = WINDOW * u[peak]
    factors = np.array([calibration_factor(u, r_traction, w_exact, i, window) for i in range(peak + 1)])
    r_reconstructed = r_traction[:peak + 1] / factors
    if has_residual:
        r_total = np.abs(data[:, COL_RES_L] + data[:, COL_RES_R])
        peak_consistent = first_peak_index(r_total)
        r_peak = r_total[peak_consistent]
        curve_u, curve_r = u, r_total
        w_peak = abs(data[peak_consistent, COL_W_RES])
        shape = r_total
    else:
        peak_consistent = int(np.nanargmax(r_reconstructed))
        r_peak = r_reconstructed[peak_consistent]
        curve_u, curve_r = u[:peak + 1], r_reconstructed
        w_peak = w_exact[peak_consistent]
        shape = r_traction
    shape_peak = first_peak_index(shape)
    last = len(shape) - 1
    elastic = int(np.argmin(np.abs(u - 0.003)))
    return {
        "source": "residual columns" if has_residual else "energy balance",
        "Ry_total_peak": r_peak,
        "uy_peak": u[peak_consistent],
        "max_over_first_peak": float(np.max(shape) / shape[shape_peak]),
        "min_after_first_peak": float(np.min(shape[shape_peak:]) / shape[shape_peak]),
        "ended_rising": bool(int(np.argmax(shape)) == last and last > shape_peak),
        "uy_final": u[-1],
        "steps": len(u),
        "W_peak": w_peak,
        "Pi_el_peak": data[peak_consistent, COL_PI_EL],
        "Pi_frac_peak": data[peak_consistent, COL_PI_FRAC],
        "D_s_peak": data[peak_consistent, COL_D_S],
        "factor_elastic": r_traction[elastic] * u[elastic] / (2.0 * data[elastic, COL_PI_EL]),
        "factor_peak": factors[peak],
        "old_Ry_left_peak": r_left[peak_left],
        "old_Ry_total_peak": r_traction[peak],
        "old_uy_peak": u[peak_left],
        "old_W_strips_peak": abs(data[peak_left, COL_W_STRIPS]),
        "old_W_boundary_peak": abs(data[peak_left, COL_W_BOUNDARY]),
        "_curve": (curve_u, curve_r, peak_consistent),
        "_old_curve": (u, r_traction, peak),
    }


def percent(a, b):
    return 100.0 * (a / b - 1.0)


def build_comparisons(results):
    rows = []
    conditions = sorted({(m, beta, eps) for (m, beta, eps, _, _) in results})
    for mobility, beta, epsilon in conditions:
        for rho, case, reference in COMPARISONS:
            a = results.get((mobility, beta, epsilon, rho, case))
            # E_min / E_max do not depend on beta_phi: fall back to the beta_phi = 0.01 reference
            b = results.get((mobility, beta, epsilon, rho, reference)) or results.get((mobility, 0.01, epsilon, rho, reference))
            if a is None or b is None:
                continue
            rows.append({
                "M": mobility, "beta_phi": beta, "beta_s": epsilon, "rho": rho,
                "comparison": f"{CASE_TEXT[case]} vs {CASE_TEXT[reference]}",
                "Ry_published_pct": percent(a["old_Ry_left_peak"], b["old_Ry_left_peak"]),
                "Ry_consistent_pct": percent(a["Ry_total_peak"], b["Ry_total_peak"]),
                "W_strips_published_pct": percent(a["old_W_strips_peak"], b["old_W_strips_peak"]),
                "W_boundary_pct": percent(a["old_W_boundary_peak"], b["old_W_boundary_peak"]),
                "W_consistent_pct": percent(a["W_peak"], b["W_peak"]),
            })
    return rows


RUN_FIELDS = ["M", "beta_phi", "beta_s", "rho", "case", "source", "Ry_total_peak", "uy_peak", "max_over_first_peak",
              "min_after_first_peak", "ended_rising", "uy_final", "steps", "W_peak", "Pi_el_peak",
              "Pi_frac_peak", "D_s_peak", "factor_elastic", "factor_peak", "old_Ry_left_peak", "old_Ry_total_peak",
              "old_uy_peak", "old_W_strips_peak", "old_W_boundary_peak"]


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items() if k in fields})


def run_rows(results):
    rows = []
    for (mobility, beta, epsilon, rho, case), r in sorted(results.items()):
        rows.append({"M": mobility, "beta_phi": beta, "beta_s": epsilon, "rho": rho, "case": CASE_TEXT[case], **r})
    return rows


def write_markdown(path, results, comparisons):
    lines = ["# Energy-consistent peak metrics (from existing histories)", "",
             "Generated by `14_energy_consistent_metrics.py`. R_y is the total of both strips in N/mm, W in Nmm/mm.",
             "\"published\" = traction integrals as used in the manuscript (R_y: left strip only, column 2; "
             "W at peak: column 4, two strips). \"consistent\" = from the energy balance (or residual columns).",
             "Peak = first peak of the total reaction (first load drop of more than 1 %).", ""]
    for mobility in sorted({k[0] for k in results}):
        lines += [f"## M = {mobility:g}", "",
                  "| beta_phi | beta_s | structure | R_y total at first peak | max / first peak | min after first peak / first peak | u_y at peak | final u_y | W at peak | 2 x R_y left, traction (old measure) | W strips (col 4) | W boundary (col 13) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for (m, beta, epsilon, rho, case), r in sorted(results.items()):
            if m != mobility:
                continue
            rising = " (still rising at end)" if r["ended_rising"] else ""
            lines.append(f"| {beta:g} | {epsilon:g} | rho = {rho:g} {CASE_TEXT[case]} | {r['Ry_total_peak']:.2f} | "
                         f"{r['max_over_first_peak']:.3f}{rising} | {r['min_after_first_peak']:.3f} | {r['uy_peak']:.5f} | {r['uy_final']:.5f} | "
                         f"{r['W_peak']:.4f} | {2 * r['old_Ry_left_peak']:.2f} | {r['old_W_strips_peak']:.4f} | {r['old_W_boundary_peak']:.4f} |")
        lines += ["", "| beta_phi | beta_s | comparison | max R_y published | max R_y consistent | W at peak published (col 4) | W at peak col 13 | W at peak consistent |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in comparisons:
            if c["M"] != mobility:
                continue
            lines.append(f"| {c['beta_phi']:g} | {c['beta_s']:g} | rho = {c['rho']:g}: {c['comparison']} | {c['Ry_published_pct']:+.1f} % | "
                         f"{c['Ry_consistent_pct']:+.1f} % | {c['W_strips_published_pct']:+.1f} % | {c['W_boundary_pct']:+.1f} % | {c['W_consistent_pct']:+.1f} % |")
        lines.append("")
    Path(path).write_text("\n".join(lines))


def style_axes(ax):
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.grid(True, alpha=0.3)


def save_figure(fig, output_dir, stem):
    legends = [child for ax in fig.axes for child in ax.get_children() if isinstance(child, matplotlib.legend.Legend)]
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight", bbox_extra_artists=legends)
    plt.close(fig)


def structure_handles():
    return [Line2D([0], [0], color=ENERGY_DATASET_COLORS[(rho, case)], linewidth=2.4,
                   label=rf"$\rho={rho:g}$, {CASE_LABELS[case]}") for rho, case in STRUCTURES]


def plot_metric_vs_beta_s(results, mobility, beta, output_dir, stem, ylabel, series):
    """series: list of (label, key or callable, linestyle, marker, alpha)."""
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    found = False
    for rho, case in STRUCTURES:
        eps_values = sorted(k[2] for k in results if k[0] == mobility and k[1] == beta and k[3] == rho and k[4] == case)
        if len(eps_values) < 2:
            continue
        found = True
        for _, getter, linestyle, marker, alpha in series:
            values = [getter(results[(mobility, beta, eps, rho, case)]) for eps in eps_values]
            ax.plot(eps_values, values, color=ENERGY_DATASET_COLORS[(rho, case)], linestyle=linestyle, marker=marker,
                    markersize=9, linewidth=2.2, alpha=alpha)
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel(r"$\beta_s$ in mm")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    method_handles = [Line2D([0], [0], color="#2f2f2f", linestyle=ls, marker=mk, markersize=8, linewidth=2.2, alpha=al, label=lab)
                      for lab, _, ls, mk, al in series]
    first = ax.legend(handles=structure_handles(), fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=method_handles, fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save_figure(fig, output_dir, stem)


def plot_curves(results, mobility, beta, epsilon, output_dir, stem):
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    found = False
    for rho, case in STRUCTURES:
        r = results.get((mobility, beta, epsilon, rho, case))
        if r is None:
            continue
        found = True
        color = ENERGY_DATASET_COLORS[(rho, case)]
        if r["source"] == "energy balance":
            u_old, r_old, _ = r["_old_curve"]
            ax.plot(u_old, r_old, color=color, linestyle=(0, (4.0, 2.5)), linewidth=1.6, alpha=0.75)
        u_new, r_new, peak = r["_curve"]
        valid = np.isfinite(r_new)
        ax.plot(u_new[valid], r_new[valid], color=color, linewidth=2.4)
        ax.scatter(u_new[peak], r_new[peak], s=95, marker="X", color=color, edgecolor="#1f1f1f", linewidth=0.85, zorder=5)
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(r"total $R_y$ in N/mm")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    methods = [Line2D([0], [0], color="#2f2f2f", linewidth=2.4, label="energy-consistent (to peak)"),
               Line2D([0], [0], color="#2f2f2f", linestyle=(0, (4.0, 2.5)), linewidth=1.6, alpha=0.75, label="traction integral"),
               Line2D([0], [0], marker="X", color="#2f2f2f", linewidth=0.0, markersize=9, label=r"$\max R_y$")]
    first = ax.legend(handles=structure_handles(), fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=methods, fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save_figure(fig, output_dir, stem)


def mobility_label(mobility):
    exponent = np.log10(mobility)
    return rf"$M=10^{{{exponent:.0f}}}$" if abs(exponent - round(exponent)) < 1e-9 else rf"$M={mobility:g}$"


def plot_mobility_comparison_vs_beta_s(results, mobilities, beta, output_dir, stem, key, ylabel):
    styles = ["-", (0, (4.0, 2.5)), (0, (1.2, 2.2))]
    markers = ["o", "s", "^"]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    found = False
    for rho, case in STRUCTURES:
        for mobility, linestyle, marker in zip(mobilities, styles, markers):
            eps_values = sorted(k[2] for k in results if k[0] == mobility and k[1] == beta and k[3] == rho and k[4] == case)
            if len(eps_values) < 2:
                continue
            found = True
            ax.plot(eps_values, [results[(mobility, beta, eps, rho, case)][key] for eps in eps_values],
                    color=ENERGY_DATASET_COLORS[(rho, case)], linestyle=linestyle, marker=marker, markersize=9,
                    linewidth=2.2, alpha=1.0 if mobility == mobilities[0] else 0.7)
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel(r"$\beta_s$ in mm")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    handles = [Line2D([0], [0], color="#2f2f2f", linestyle=ls, marker=mk, markersize=8, linewidth=2.2, label=mobility_label(m))
               for m, ls, mk in zip(mobilities, styles, markers)]
    first = ax.legend(handles=structure_handles(), fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=handles, fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save_figure(fig, output_dir, stem)


def plot_mobility_comparison_curves(results, mobilities, beta, epsilon, output_dir, stem):
    styles = ["-", (0, (4.0, 2.5)), (0, (1.2, 2.2))]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    found = False
    for rho, case in STRUCTURES:
        for mobility, linestyle in zip(mobilities, styles):
            r = results.get((mobility, beta, epsilon, rho, case))
            if r is None:
                continue
            found = True
            color = ENERGY_DATASET_COLORS[(rho, case)]
            u_new, r_new, peak = r["_curve"]
            valid = np.isfinite(r_new)
            ax.plot(u_new[valid], r_new[valid], color=color, linestyle=linestyle, linewidth=2.2,
                    alpha=1.0 if mobility == mobilities[0] else 0.7)
            ax.scatter(u_new[peak], r_new[peak], s=95, marker="X", color=color, edgecolor="#1f1f1f", linewidth=0.85, zorder=5)
    if not found:
        plt.close(fig)
        return
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(r"total $R_y$ in N/mm")
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    style_axes(ax)
    handles = [Line2D([0], [0], color="#2f2f2f", linestyle=ls, linewidth=2.2, label=mobility_label(m))
               for m, ls in zip(mobilities, styles)]
    handles.append(Line2D([0], [0], marker="X", color="#2f2f2f", linewidth=0.0, markersize=9, label="first peak"))
    first = ax.legend(handles=structure_handles(), fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(first)
    ax.legend(handles=handles, fontsize=LEGEND_FONT_SIZE, loc="lower left", bbox_to_anchor=(1.01, 0.0))
    save_figure(fig, output_dir, stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_root", nargs="?", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--extra-root", type=Path, action="append", default=[],
                        help="additional result roots, e.g. the mobility study (files with _M<label>)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--a-value", type=int, default=6)
    parser.add_argument("--split", default="spectral")
    parser.add_argument("--no-usetex", action="store_true")
    args = parser.parse_args()

    plt.rcParams.update({"text.usetex": not args.no_usetex, "font.family": "serif",
                         "font.serif": ["Computer Modern Roman"], "mathtext.fontset": "cm",
                         "axes.unicode_minus": False, "font.size": 15})

    paths = {}
    for root in [args.result_root, *args.extra_root]:
        found = discover(root, args.a_value, args.split)
        print(f"[INFO] {len(found)} histories below {root}")
        paths.update(found)
    if not paths:
        raise SystemExit("No result histories found")
    results = {}
    for key, path in paths.items():
        data = np.atleast_2d(np.loadtxt(path, comments="#"))
        if data.shape[0] < 5 or np.max(np.abs(data[:, COL_RY_L])) <= 0.0:
            print(f"[INFO] skipping {path.name}: only {data.shape[0]} steps (run in progress?)")
            continue
        results[key] = evaluate(path)
    comparisons = build_comparisons(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "energy_consistent_runs.csv", run_rows(results), RUN_FIELDS)
    write_csv(args.output_dir / "energy_consistent_comparisons.csv", comparisons, list(comparisons[0].keys()))
    write_markdown(args.output_dir / "energy_consistent_summary.md", results, comparisons)

    for mobility in sorted({k[0] for k in results}):
        tag = "" if mobility == REFERENCE_MOBILITY else f"_M{mobility:g}"
        for beta in sorted({k[1] for k in results if k[0] == mobility}):
            beta_tag = f"beta{str(beta).replace('.', '_')}"
            plot_metric_vs_beta_s(
                results, mobility, beta, args.output_dir, f"max_Ry_total_vs_beta_s_{beta_tag}{tag}",
                r"total $\max R_y$ in N/mm",
                [("energy-consistent", lambda r: r["Ry_total_peak"], "-", "o", 1.0),
                 ("traction integral", lambda r: r["old_Ry_total_peak"], (0, (4.0, 2.5)), "s", 0.75)])
            plot_metric_vs_beta_s(
                results, mobility, beta, args.output_dir, f"W_at_peak_vs_beta_s_{beta_tag}{tag}",
                r"$W(R_y=\max R_y)$ in Nmm/mm",
                [("energy-consistent", lambda r: r["W_peak"], "-", "o", 1.0),
                 ("traction, two strips", lambda r: r["old_W_strips_peak"], (0, (4.0, 2.5)), "s", 0.75),
                 ("traction, whole boundary", lambda r: r["old_W_boundary_peak"], (0, (1.2, 2.2)), "^", 0.75)])
            for epsilon in sorted({k[2] for k in results if k[0] == mobility and k[1] == beta}):
                plot_curves(results, mobility, beta, epsilon, args.output_dir,
                            f"Ry_total_vs_uy_{beta_tag}_eps{str(epsilon).replace('.', '_')}{tag}")
    # highest mobility against the published M = 100 (energy-consistent values for both)
    all_mobilities = sorted({k[0] for k in results}, reverse=True)
    if len(all_mobilities) > 1 and REFERENCE_MOBILITY in all_mobilities:
        pair = [all_mobilities[0], REFERENCE_MOBILITY]
        for beta in sorted({k[1] for k in results if k[0] == pair[0]}):
            beta_tag = f"beta{str(beta).replace('.', '_')}"
            plot_mobility_comparison_vs_beta_s(results, pair, beta, args.output_dir, f"max_Ry_total_vs_beta_s_{beta_tag}_M_comparison",
                                               "Ry_total_peak", r"total $R_y$ at first peak in N/mm")
            plot_mobility_comparison_vs_beta_s(results, pair, beta, args.output_dir, f"W_at_peak_vs_beta_s_{beta_tag}_M_comparison",
                                               "W_peak", r"$W$ at first peak in Nmm/mm")
            for epsilon in sorted({k[2] for k in results if k[0] == pair[0] and k[1] == beta}):
                plot_mobility_comparison_curves(results, pair, beta, epsilon, args.output_dir,
                                                f"Ry_total_vs_uy_{beta_tag}_eps{str(epsilon).replace('.', '_')}_M_comparison")
    print(f"[INFO] wrote {len(results)} runs and {len(comparisons)} comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
