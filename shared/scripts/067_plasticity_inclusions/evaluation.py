#!/usr/bin/env python3
"""Generate paper plots for the 067 plasticity-inclusions study.

The graph files written by ``run_simulation.py`` use the same column order as
the earlier plasticity paper:

0 pseudo-time, 1 J_x, 2 J_y, 3 crack-tip position x_ct, 4 surfing BC position.
The remaining columns are diagnostic quantities and are not needed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "manuscript"
PICS = MANUSCRIPT / "pics"
REPRESENTATIVE = RESULTS / "simulation_20260511_152054_WSTEG1.0_KINC0.5_GCINC0.5"
EXCLUDED_WSTEGS = {0.75}
MAX_WSTEG = 2.0
# Evaluate the repeated interaction pattern spanning inclusions 4 through 6.
EVALUATED_PATTERN_FIRST_INCLUSION = 4

COL_T = 0
COL_JX = 1
COL_XCT = 3
COL_XBC = 4

CASE_RE = re.compile(r"WSTEG(?P<wsteg>[0-9.]+)_KINC(?P<kinc>[0-9.]+)_GCINC(?P<gcinc>[0-9.]+)")


@dataclass(frozen=True)
class Case:
    folder: Path
    wsteg: float
    kinc: float
    gcinc: float
    data: np.ndarray
    lam_eff: float
    mu_eff: float
    mesh_hmin: float
    mesh_hmean: float
    dinclusion: float = 1.0
    nholes: int = 6

    @property
    def jeff(self) -> float:
        data = data_for_effective_resistance(self)
        if len(data) == 0:
            data = characteristic_section(self)
        if len(data) == 0:
            data = self.data
        return float(np.nanmax(data[:, COL_JX]))


def read_parameters(path: Path) -> dict[str, float | str]:
    parameters: dict[str, float | str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        try:
            parameters[key] = float(value)
        except ValueError:
            parameters[key] = value
    return parameters


def parse_case(folder: Path) -> tuple[float, float, float] | None:
    match = CASE_RE.search(folder.name)
    if not match:
        return None
    return tuple(float(match.group(name)) for name in ("wsteg", "kinc", "gcinc"))


def load_cases() -> list[Case]:
    cases: list[Case] = []
    for folder in sorted(RESULTS.glob("simulation_*")):
        parsed = parse_case(folder)
        graph = folder / "run_simulation_graphs.txt"
        params = folder / "parameters.txt"
        if parsed is None or not graph.exists() or not params.exists():
            continue
        data = np.loadtxt(graph, comments="#")
        if data.ndim != 2 or data.shape[1] <= COL_XBC:
            continue
        p = read_parameters(params)
        if parsed[0] > MAX_WSTEG or any(np.isclose(parsed[0], excluded) for excluded in EXCLUDED_WSTEGS):
            continue
        cases.append(
            Case(
                folder=folder,
                wsteg=parsed[0],
                kinc=parsed[1],
                gcinc=parsed[2],
                data=data,
                lam_eff=float(p.get("lam_effective", np.nan)),
                mu_eff=float(p.get("mue_effective", np.nan)),
                mesh_hmin=float(p.get("min_edge_length", np.nan)),
                mesh_hmean=float(p.get("mean_edge_length", np.nan)),
                dinclusion=float(p.get("dinclusion", p.get("dhole", 1.0))),
                nholes=int(float(p.get("nholes", 6))),
            )
        )
    if not cases:
        raise RuntimeError(f"No readable simulation cases found below {RESULTS}")
    return cases


def savefig(path: Path, *, tight: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=350)
    plt.close()
    print(path.relative_to(ROOT))


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="0.88", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def label_factor(prefix: str, value: float, suffix: str = "") -> str:
    return rf"${prefix}={value:g}{suffix}$"


def subset(cases: list[Case], *, wsteg: float | None = None, kinc: float | None = None, gcinc: float | None = None) -> list[Case]:
    out = cases
    if wsteg is not None:
        out = [c for c in out if np.isclose(c.wsteg, wsteg)]
    if kinc is not None:
        out = [c for c in out if np.isclose(c.kinc, kinc)]
    if gcinc is not None:
        out = [c for c in out if np.isclose(c.gcinc, gcinc)]
    return sorted(out, key=lambda c: (c.wsteg, c.kinc, c.gcinc))


def inclusion_start_positions(case: Case) -> list[float]:
    """Match the middle-inclusion coordinate convention used in KLAIM_2025."""
    cell_width = case.dinclusion + case.wsteg
    starts = []
    for n in range(case.nholes):
        center = 1.5 * cell_width + n * cell_width
        starts.append(center - 0.5 * case.dinclusion)
    return starts


def characteristic_bounds(case: Case, start_inclusion: int | None = None) -> tuple[float, float]:
    if start_inclusion is None:
        start_inclusion = EVALUATED_PATTERN_FIRST_INCLUSION
    return effective_resistance_bounds(case, start_inclusion)


def effective_resistance_bounds(case: Case, start_inclusion: int | None = None) -> tuple[float, float]:
    # Use the same inclusion-position convention as 046_plasticity.  The
    # selected characteristic record spans the central inclusion interaction
    # section.  A virtual downstream start position is permitted so the final
    # inclusion section can be evaluated for start_inclusion == nholes.
    if start_inclusion is None:
        start_inclusion = EVALUATED_PATTERN_FIRST_INCLUSION
    cell_width = case.dinclusion + case.wsteg
    if start_inclusion < 1 or start_inclusion > case.nholes:
        raise ValueError(
            f"Cannot evaluate inclusion section {start_inclusion} for {case.nholes} inclusions"
        )

    def start_position(index: int) -> float:
        center = 1.5 * cell_width + index * cell_width
        return center - 0.5 * case.dinclusion

    low = start_position(start_inclusion - 2) - 0.5 * case.wsteg
    high = start_position(start_inclusion) - 0.5 * case.wsteg
    if np.isclose(case.gcinc, 1.0) and np.isclose(case.kinc, 1.5):
        low += case.wsteg
    return low, high


def filter_by_xct(case: Case, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    mask = (case.data[:, COL_XCT] >= low) & (case.data[:, COL_XCT] <= high)
    return case.data[mask]


def characteristic_section(case: Case) -> np.ndarray:
    return filter_by_xct(case, characteristic_bounds(case))


def data_for_effective_resistance(case: Case) -> np.ndarray:
    return filter_by_xct(case, effective_resistance_bounds(case))


def plot_crack_tip(cases: list[Case]) -> None:
    case = subset(cases, wsteg=1.0, kinc=1.0, gcinc=1.0)[0]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(case.data[:, COL_T], case.data[:, COL_XCT], label=r"$x_{\rm ct}$", lw=1.8)
    ax.plot(case.data[:, COL_T], case.data[:, COL_XBC], label=r"$x_{\rm bc}$", lw=1.8)
    ax.set_xlabel(r"$t/[L/\dot{x}_{\rm bc}]$")
    ax.set_ylabel(r"position $/L$")
    ax.legend(frameon=False)
    style_axes(ax)
    savefig(PICS / "fig09_crack_tip_tracking.png")


def plot_fig10_selected_cases(cases: list[Case]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    selected_cases = [
        (1.0, 1.0, "#333333", "reference"),
        (1.5, 0.5, "#D55E00", "stiff, weak"),
        (0.5, 1.0, "#0072B2", "compliant"),
    ]
    for kinc, ginc, color, description in selected_cases:
        case = subset(cases, wsteg=1.0, kinc=kinc, gcinc=ginc)[0]
        ax.plot(
            case.data[:, COL_XCT],
            case.data[:, COL_JX],
            lw=1.6,
            color=color,
            label=rf"{description} $({kinc:g},{ginc:g})$",
        )
    ax.set_xlabel(r"crack-tip position $x_{\rm ct}/L$")
    ax.set_ylabel(r"$J_x/G_c^0$")
    ax.set_xlim(0, 15)
    ax.legend(
        frameon=False,
        title=r"$w_s=1.0L,\quad(k_{\mathrm{inc}},g_{\mathrm{inc}})$",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        fontsize=8,
        title_fontsize=9,
    )
    style_axes(ax)
    savefig(PICS / "fig10_Jx_xct_selected_cases.png")


def plot_jintegral_sections(cases: list[Case], gcinc: float, figure_no: int) -> None:
    """Plot J_x(x_ct) sections for one inclusion toughness and all stiffnesses."""
    wstegs = sorted({c.wsteg for c in cases})
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":"]
    color_by_wsteg = dict(zip(wstegs, colors))
    marker_by_wsteg = dict(zip(wstegs, markers))
    linestyle_by_wsteg = {w: linestyles[i // len(colors) % len(linestyles)] for i, w in enumerate(wstegs)}

    fig, axes = plt.subplots(3, 1, figsize=(6.7, 8.0), sharex=True, sharey=True)
    for ax, kinc in zip(axes, [0.5, 1.0, 1.5]):
        family = subset(cases, kinc=kinc, gcinc=gcinc)
        for case in family:
            data = characteristic_section(case)
            if len(data) == 0:
                continue
            ax.plot(
                data[:, COL_XCT],
                data[:, COL_JX],
                lw=1.2,
                color=color_by_wsteg[case.wsteg],
                linestyle=linestyle_by_wsteg[case.wsteg],
                marker=marker_by_wsteg[case.wsteg],
                markersize=3.0,
                markevery=max(len(data) // 35, 1),
                label=label_factor(r"w_s/L", case.wsteg),
            )
        ax.set_title(label_factor(r"k_{\mathrm{inc}}", kinc))
        ax.set_ylabel(r"$J_x/G_c^0$")
        style_axes(ax)
    if np.isclose(gcinc, 1.5):
        axes[0].set_ylim(1.3, 2.0)
    axes[-1].set_xlabel(r"crack-tip position $x_{\rm ct}/L$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=4,
        loc="lower center",
        frameon=False,
        title=r"ligament width",
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(label_factor(r"g_{\mathrm{inc}}", gcinc), y=0.995)
    fig.subplots_adjust(top=0.94, bottom=0.15, hspace=0.28)
    savefig(PICS / f"fig{figure_no}_Jx_sections_Gcinc_{str(gcinc).replace('.', 'p')}.png", tight=False)


def plot_jeff_families(cases: list[Case]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.1), sharey=True)
    for ax, gcinc in zip(axes, [0.5, 1.0, 1.5]):
        for kinc, color in [(0.5, "#0072B2"), (1.0, "#333333"), (1.5, "#D55E00")]:
            family = subset(cases, kinc=kinc, gcinc=gcinc)
            ax.plot(
                [c.wsteg for c in family],
                [c.jeff for c in family],
                marker="o",
                lw=1.7,
                color=color,
                label=label_factor(r"k_{\mathrm{inc}}", kinc),
            )
        ax.set_title(label_factor(r"g_{\mathrm{inc}}", gcinc))
        ax.set_xlabel(r"wall width $w_s/L$")
        ax.set_xscale("log")
        ax.set_xticks(sorted({c.wsteg for c in cases}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        style_axes(ax)
    axes[0].set_ylabel(r"$J_{x,c}^{\rm eff}/G_c^0$")
    axes[-1].legend(frameon=False)
    savefig(PICS / "fig15_Jeff_wsteg_all_Gcinc.png")


def plot_combined_heatmap(cases: list[Case]) -> None:
    df = pd.DataFrame(
        {
            "wsteg": [c.wsteg for c in cases],
            "kinc": [c.kinc for c in cases],
            "gcinc": [c.gcinc for c in cases],
            "jeff": [c.jeff for c in cases],
        }
    )
    ref = df[np.isclose(df.kinc, 1.0) & np.isclose(df.gcinc, 1.0)].set_index("wsteg")["jeff"]
    df["relative"] = [row.jeff / ref.loc[row.wsteg] for row in df.itertuples()]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.65), sharey=True)
    vmin = float(df["relative"].min())
    vmax = float(df["relative"].max())
    for panel, ax, wsteg in zip(["a)", "b)", "c)"], axes, [0.5, 1.0, 2.0]):
        pivot = df[np.isclose(df.wsteg, wsteg)].pivot(index="gcinc", columns="kinc", values="relative")
        image = ax.imshow(pivot.values, origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(rf"$w_s={wsteg:g}L$", fontsize=15, pad=10)
        ax.text(-0.20, 1.10, panel, transform=ax.transAxes, fontsize=19, fontweight="bold")
        ax.set_xticks(range(len(pivot.columns)), [f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [f"{v:g}" for v in pivot.index])
        ax.tick_params(labelsize=14)
        ax.set_xlabel(r"$k_{\mathrm{inc}}$", fontsize=15)
        if ax is axes[0]:
            ax.set_ylabel(r"$g_{\mathrm{inc}}$", fontsize=15)
        for i, gc in enumerate(pivot.index):
            for j, k in enumerate(pivot.columns):
                value = pivot.loc[gc, k]
                color = "white" if value < 0.88 else "#161616"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=17)
    fig.subplots_adjust(left=0.08, right=0.82, top=0.84, bottom=0.20, wspace=0.30)
    cbar_ax = fig.add_axes([0.87, 0.22, 0.028, 0.56])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(
        r"$J_{x,c}^{\rm eff}/J_{x,c}^{\rm eff}(k_{\mathrm{inc}}=1,g_{\mathrm{inc}}=1)$",
        fontsize=13,
    )
    savefig(PICS / "fig14_combined_parameter_contribution.png", tight=False)


def render_material_schematic() -> None:
    draw_material_schematic(PICS / "material_layout_inclusions.png")


def material_layout_geometry() -> tuple[float, float, float, float, float, float, float, float, int]:
    params = read_parameters(REPRESENTATIVE / "parameters.txt")
    wsteg = float(params["wsteg"])
    d = float(params["dinclusion"])
    nholes = int(params["nholes"])
    wc = wsteg + d
    h5_path = REPRESENTATIVE / "run_simulation.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as h5:
            points = h5["/Mesh/Grid/geometry"][:]
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
    else:
        xmin, xmax, ymin, ymax = 0.0, 16.0, -10.0, 10.0

    region_width = nholes * wc
    region_height = 3 * wc
    x0 = 2.0
    y0 = -0.5 * region_height
    return xmin, xmax, ymin, ymax, x0, y0, wc, d, nholes


def add_material_layout(ax: plt.Axes, *, alpha: float = 1.0) -> None:
    xmin, xmax, ymin, ymax, x0, y0, wc, d, nholes = material_layout_geometry()
    region_width = nholes * wc
    region_height = 3 * wc
    ax.add_patch(
        plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            color="#DCE8F2",
            ec="0.25",
            lw=0.8,
            alpha=alpha,
            zorder=0,
        )
    )
    ax.add_patch(
        plt.Rectangle(
            (x0, y0),
            region_width,
            region_height,
            color="#F3E4D4",
            ec="0.25",
            lw=0.8,
            alpha=alpha,
            zorder=1,
        )
    )
    for row in range(3):
        for col in range(nholes):
            cx = x0 + (col + 0.5) * wc
            cy = y0 + (row + 0.5) * wc
            ax.add_patch(
                plt.Circle(
                    (cx, cy),
                    d / 2,
                    fc="#5B8DB8",
                    ec="#17324D",
                    lw=0.8,
                    alpha=alpha,
                    zorder=2,
                )
            )


def finalize_domain_axes(ax: plt.Axes) -> None:
    xmin, xmax, ymin, ymax, *_ = material_layout_geometry()
    ax.set_aspect("equal")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_axis_off()


def draw_material_schematic(output_path: Path) -> None:
    xmin, xmax, ymin, ymax, *_ = material_layout_geometry()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    add_material_layout(ax)
    finalize_domain_axes(ax)
    savefig(output_path)


def h5_dataset_group_keys(h5: h5py.File, name: str) -> list[str]:
    group = h5[f"/Function/{name}"]
    return sorted(group.keys(), key=lambda key: float(key.replace("_", ".", 1)))


def render_hdf_snapshot() -> None:
    h5_path = REPRESENTATIVE / "run_simulation.h5"
    if not h5_path.exists():
        return
    with h5py.File(h5_path, "r") as h5:
        points = h5["/Mesh/Grid/geometry"][:]
        triangles = h5["/Mesh/Grid/topology"][:]
        triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        add_material_layout(ax, alpha=0.78)
        ax.triplot(triang, color="#263746", lw=0.12, alpha=0.80, zorder=3)
        finalize_domain_axes(ax)
        savefig(PICS / "mesh_representative.png")

        if "/Function/s" in h5:
            key = h5_dataset_group_keys(h5, "s")[-1]
            s = np.asarray(h5[f"/Function/s/{key}"][:]).reshape(-1)
            fig, ax = plt.subplots(figsize=(7.8, 4.4))
            add_material_layout(ax, alpha=0.95)
            plot = ax.tripcolor(
                triang,
                s,
                shading="gouraud",
                cmap="coolwarm",
                vmin=0.0,
                vmax=1.0,
                alpha=0.68,
                zorder=3,
            )
            finalize_domain_axes(ax)
            cbar = fig.colorbar(plot, ax=ax, shrink=0.75, pad=0.01)
            cbar.set_label(r"$s$", fontsize=14)
            cbar.ax.tick_params(labelsize=11)
            savefig(PICS / "failure_pattern_phasefield.png")


def write_summary(cases: list[Case]) -> None:
    df = pd.DataFrame(
        {
            "wsteg": [c.wsteg for c in cases],
            "kinc": [c.kinc for c in cases],
            "gcinc": [c.gcinc for c in cases],
            "jeff": [c.jeff for c in cases],
            "lam_eff": [c.lam_eff for c in cases],
            "mu_eff": [c.mu_eff for c in cases],
            "mesh_hmin": [c.mesh_hmin for c in cases],
            "mesh_hmean": [c.mesh_hmean for c in cases],
        }
    ).sort_values(["gcinc", "kinc", "wsteg"])
    df.to_csv(MANUSCRIPT / "data" / "summary_effective_resistance.csv", index=False)

    baseline = df[np.isclose(df.kinc, 1.0) & np.isclose(df.gcinc, 1.0)].sort_values("wsteg")
    low_gc = df[np.isclose(df.kinc, 1.0) & np.isclose(df.gcinc, 0.5)].set_index("wsteg")["jeff"]
    high_gc = df[np.isclose(df.kinc, 1.0) & np.isclose(df.gcinc, 1.5)].set_index("wsteg")["jeff"]
    low_k = df[np.isclose(df.kinc, 0.5) & np.isclose(df.gcinc, 1.0)].set_index("wsteg")["jeff"]
    high_k = df[np.isclose(df.kinc, 1.5) & np.isclose(df.gcinc, 1.0)].set_index("wsteg")["jeff"]

    def pct(a: pd.Series, b: pd.Series) -> float:
        values = ((a - b) / b * 100.0).dropna()
        return float(values.mean())

    macros = rf"""
\newcommand{{\NumCases}}{{{len(df)}}}
\newcommand{{\WstegValues}}{{{", ".join(f"{v:g}" for v in sorted(df.wsteg.unique()))}}}
\newcommand{{\JeffBaselineMin}}{{{baseline.jeff.min():.2f}}}
\newcommand{{\JeffBaselineMax}}{{{baseline.jeff.max():.2f}}}
\newcommand{{\MeanGcLowEffect}}{{{pct(low_gc, baseline.set_index("wsteg")["jeff"]):+.1f}\%}}
\newcommand{{\MeanGcHighEffect}}{{{pct(high_gc, baseline.set_index("wsteg")["jeff"]):+.1f}\%}}
\newcommand{{\MeanKLowEffect}}{{{pct(low_k, baseline.set_index("wsteg")["jeff"]):+.1f}\%}}
\newcommand{{\MeanKHighEffect}}{{{pct(high_k, baseline.set_index("wsteg")["jeff"]):+.1f}\%}}
"""
    (MANUSCRIPT / "generated_results.tex").write_text(macros.strip() + "\n")


def main() -> None:
    PICS.mkdir(parents=True, exist_ok=True)
    (MANUSCRIPT / "data").mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    write_summary(cases)
    plot_crack_tip(cases)
    plot_fig10_selected_cases(cases)
    for number, gcinc in zip([11, 12, 13], [0.5, 1.0, 1.5]):
        plot_jintegral_sections(cases, gcinc=gcinc, figure_no=number)
    plot_combined_heatmap(cases)
    plot_jeff_families(cases)
    render_material_schematic()
    render_hdf_snapshot()
    print(f"Processed {len(cases)} simulation cases.")


if __name__ == "__main__":
    main()
