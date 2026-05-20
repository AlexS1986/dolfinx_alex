#!/usr/bin/env python3
"""Create overview plots of the final phase-field variable from DOLFINx XDMF/H5 results."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.tri import Triangulation

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


RESULT_RE = re.compile(
    r"^results_(?P<case>.+)_(?P<split>spectral|volumetric)(?:_eps(?P<epsilon>[0-9_]+))?\.xdmf$"
)

SIMULATION_FOLDER_RE = re.compile(
    r"^simulation_.*_SPLIT(?P<split>spectral|volumetric)_EPS(?P<epsilon>[0-9_]+)$"
)

CASE_TITLE_SIZE = 36
PAGE_TITLE_SIZE = 24
COLORBAR_LABEL_SIZE = 44
COLORBAR_TICK_SIZE = 34
COLORBAR_LABEL_PAD = 30
ROW_LABEL_SIZE = 46
POISSON_RATIO = 0.3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the final phase-field s from all results_*.xdmf files below the results tree."
    )
    parser.add_argument(
        "result_roots",
        nargs="*",
        type=Path,
        default=None,
        help="Result folders to scan recursively. Defaults to ./results next to this script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF file. Defaults to ./plots/phasefield_s_overview.pdf next to this script.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=8,
        help="Minimum number of width units per page. a=3 uses one unit, a=6 uses two.",
    )
    parser.add_argument(
        "--field",
        default="s",
        help="HDF5 function field to plot. Defaults to phase-field s.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="HDF5 function fields to plot. When set, writes one overview per field.",
    )
    parser.add_argument(
        "--collapse-epsilon-fields",
        nargs="*",
        default=["E", "gc", "sigma_c"],
        help="Fields for which only one representative epsilon per split is plotted. Defaults to E, gc, and sigma_c.",
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=0.01,
        help="Beta_phi value to plot when beta variants are omitted. Defaults to 0.01.",
    )
    parser.add_argument(
        "--include-beta-variants",
        action="store_true",
        help="Plot all beta variants instead of selecting one beta slice.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("spectral", "volumetric"),
        default=None,
        help="Optional split names to include.",
    )
    parser.add_argument(
        "--a-values",
        nargs="+",
        type=int,
        default=None,
        help="Optional a values to include.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap for the contour plot. Defaults to coolwarm, blue to red.",
    )
    parser.add_argument(
        "--hide-corner-label",
        action="store_true",
        help="Do not print the beta/epsilon annotation in the top right corner.",
    )
    parser.add_argument(
        "--hide-page-title",
        action="store_true",
        help="Do not print the overview title above each page.",
    )
    parser.add_argument(
        "--hide-a-label",
        action="store_true",
        help="Do not print the a value in each subplot title.",
    )
    parser.add_argument(
        "--target-time",
        type=float,
        default=None,
        help="Optional time in seconds. The closest stored time is used for every plotted field.",
    )
    return parser.parse_args()


def case_sort_key(path: Path) -> tuple[str, float, int, float, float, float, str]:
    match = RESULT_RE.match(path.name)
    if not match:
        return (path.name, math.inf, math.inf, math.inf, math.inf, "")
    case = match.group("case")
    split = match.group("split")
    epsilon = parse_epsilon(path, match)
    mode = case.rsplit("_", 1)[-1]
    mode_order = {"min": 0, "max": 1, "vary": 2}.get(mode, 99)
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")
    return (split, epsilon, mode_order, beta, a, rho, case)


def parse_token_number(token: str | None) -> float:
    if token is None:
        return math.inf
    return float(token.replace("_", "."))


def parse_epsilon(path: Path, result_match: re.Match[str] | None = None) -> float:
    if result_match is None:
        result_match = RESULT_RE.match(path.name)
    if result_match and result_match.group("epsilon"):
        return parse_token_number(result_match.group("epsilon"))
    for parent in path.parents:
        folder_match = SIMULATION_FOLDER_RE.match(parent.name)
        if folder_match:
            return parse_token_number(folder_match.group("epsilon"))
    return math.inf


def extract_number(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if not match:
        return math.inf
    return float(match.group(1).replace("_", "."))


def split_and_case(path: Path) -> tuple[str, str]:
    match = RESULT_RE.match(path.name)
    if not match:
        return ("unknown", path.stem)
    return (match.group("split"), match.group("case"))


def split_case_epsilon(path: Path) -> tuple[str, str, float]:
    match = RESULT_RE.match(path.name)
    if not match:
        return ("unknown", path.stem, math.inf)
    return (match.group("split"), match.group("case"), parse_epsilon(path, match))


def beta_from_path(path: Path) -> float:
    match = RESULT_RE.match(path.name)
    if not match:
        return math.inf
    return extract_number(match.group("case"), r"beta_(0(?:_\d+)?)")


def a_from_path(path: Path) -> float:
    match = RESULT_RE.match(path.name)
    if not match:
        return math.inf
    return extract_number(match.group("case"), r"_a_(\d+)")


def rho_from_path(path: Path) -> float:
    match = RESULT_RE.match(path.name)
    if not match:
        return math.inf
    return extract_number(match.group("case"), r"_rho_(0_\d+)")


def width_units_for_path(path: Path) -> int:
    a_value = a_from_path(path)
    if not math.isfinite(a_value):
        return 1
    return max(1, round(a_value / 3.0))


def layout_key_for_path(path: Path, include_beta: bool) -> tuple[float, float] | tuple[float, float, float]:
    if include_beta:
        return (beta_from_path(path), a_from_path(path), rho_from_path(path))
    return (a_from_path(path), rho_from_path(path))


def build_column_layout(paths_by_mode: dict[str, list[Path]]) -> tuple[dict[tuple, int], int]:
    all_paths = [path for mode_paths in paths_by_mode.values() for path in mode_paths]
    beta_values = {beta_from_path(path) for path in all_paths if math.isfinite(beta_from_path(path))}
    include_beta = len(beta_values) > 1
    keys = sorted({layout_key_for_path(path, include_beta) for path in all_paths})

    col_by_key = {}
    current_col = 0
    for key in keys:
        representative = next(path for path in all_paths if layout_key_for_path(path, include_beta) == key)
        col_by_key[key] = current_col
        current_col += width_units_for_path(representative)
    return col_by_key, current_col


def select_fixed_beta(xdmf_files: list[Path], requested_beta: float | None) -> float:
    beta_values = sorted({beta_from_path(path) for path in xdmf_files if math.isfinite(beta_from_path(path))})
    if not beta_values:
        raise SystemExit("No beta values found in results_*.xdmf files.")
    if requested_beta is None:
        return beta_values[0]
    for beta in beta_values:
        if math.isclose(beta, requested_beta):
            return beta
    available = ", ".join(f"{beta:g}" for beta in beta_values)
    raise SystemExit(f"Requested --fixed-beta {requested_beta:g} was not found. Available beta values: {available}")


def filter_by_beta(xdmf_files: list[Path], beta: float) -> list[Path]:
    return [path for path in xdmf_files if math.isclose(beta_from_path(path), beta)]


def filter_xdmf_files(
    xdmf_files: list[Path],
    splits: list[str] | None = None,
    a_values: list[int] | None = None,
) -> list[Path]:
    filtered_files = xdmf_files
    if splits is not None:
        split_set = set(splits)
        filtered_files = [
            path
            for path in filtered_files
            if split_and_case(path)[0] in split_set
        ]
    if a_values is not None:
        a_set = set(a_values)
        filtered_files = [
            path
            for path in filtered_files
            if int(a_from_path(path)) in a_set
        ]
    return filtered_files


def collapse_epsilon_variants(xdmf_files: list[Path]) -> list[Path]:
    selected_epsilons: dict[str, float] = {}
    for path in xdmf_files:
        split, _, epsilon = split_case_epsilon(path)
        if split not in selected_epsilons or epsilon < selected_epsilons[split]:
            selected_epsilons[split] = epsilon

    return [
        path
        for path in xdmf_files
        if math.isclose(split_case_epsilon(path)[2], selected_epsilons[split_case_epsilon(path)[0]])
    ]


def readable_label(case: str, epsilon: float = math.inf) -> str:
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")
    parts = []
    if math.isfinite(beta):
        parts.append(beta_label(beta))
    if math.isfinite(a):
        parts.append(length_label("a", a))
    if math.isfinite(rho):
        parts.append(rf"$\rho={rho:g}$")
    if math.isfinite(epsilon):
        parts.append(length_label(r"\epsilon", epsilon))
    return ", ".join(parts)


def length_label(symbol: str, value: float) -> str:
    return rf"${symbol}={value:g}\,\mathrm{{mm}}$"


def beta_label(value: float) -> str:
    return rf"$\beta_{{\phi}}={value:g}\,1/\mathrm{{mm}}$"


def readable_title(case: str, epsilon: float, time: float, include_a_label: bool = True, include_time: bool = True) -> str:
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")

    first_line = []
    second_line = []
    if include_a_label and math.isfinite(a):
        first_line.append(length_label("a", a))
    if math.isfinite(rho):
        second_line.append(rf"$\rho={rho:g}$")
    if include_time:
        second_line.append(rf"$t={time:.3f}\,\mathrm{{s}}$")

    lines = [", ".join(line) for line in (first_line, second_line) if line]
    return "\n".join(lines)


def page_corner_label(paths: list[Path], epsilon: float) -> str:
    beta_values = sorted({beta_from_path(path) for path in paths if math.isfinite(beta_from_path(path))})
    parts = []
    if beta_values:
        if len(beta_values) == 1:
            parts.append(beta_label(beta_values[0]))
        else:
            parts.append(rf"$\beta_{{\phi}}\in[{beta_values[0]:g},{beta_values[-1]:g}]\,1/\mathrm{{mm}}$")
    if math.isfinite(epsilon):
        parts.append(length_label(r"\epsilon", epsilon))
    return "\n".join(parts)


def overview_title(field: str, split: str, paths: list[Path], epsilon: float, include_corner_label: bool) -> str:
    title = f"Final phase-field {field_title(field)} overview: {split}"
    if not include_corner_label:
        label = page_corner_label(paths, epsilon)
        if label:
            title += "\n" + label
    return title


def field_title(field: str) -> str:
    return {
        "s": r"$s$",
        "E": r"$E$",
        "gc": r"$G_c$",
        "sigma_c": r"$\sigma_c$",
        "sig_vol": r"$\sigma_\mathrm{vol}$",
        "sig_dev": r"$\|\sigma_\mathrm{dev}\|$",
    }.get(field, rf"${field}$")


def field_label(field: str) -> str:
    return {
        "s": r"$s$",
        "E": r"$E$ in N/mm$^2$",
        "gc": r"$G_c$ in Nmm/mm$^2$",
        "sigma_c": r"$\sigma_c$ in N/mm$^2$",
        "sig_vol": r"$\sigma_\mathrm{vol}$ in N/mm$^2$",
        "sig_dev": r"$\|\sigma_\mathrm{dev}\|$ in N/mm$^2$",
    }.get(field, rf"${field}$")


def h5_path_from_xdmf(xdmf_path: Path) -> Path:
    return xdmf_path.with_suffix(".h5")


def time_key_value(time_key: str) -> float:
    return float(time_key.replace("_", "."))


def select_time_key(keys, target_time: float | None) -> str:
    if target_time is None:
        return max(keys, key=time_key_value)
    return min(keys, key=lambda key: abs(time_key_value(key) - target_time))


def load_field(
    xdmf_path: Path,
    field: str,
    target_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h5_path = h5_path_from_xdmf(xdmf_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 companion file: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        points = np.asarray(h5["/Mesh/mesh/geometry"])
        cells = np.asarray(h5["/Mesh/mesh/topology"], dtype=np.int64)
        field_group = h5[f"/Function/{field}"]
        time_key = select_time_key(field_group.keys(), target_time)
        values = np.asarray(field_group[time_key]).reshape(-1)
        time = time_key_value(time_key)
    return points, cells, values, time


def sigma_c_values(E_values: np.ndarray, gc_values: np.ndarray, epsilon: float) -> np.ndarray:
    mu_values = E_values / (2.0 * (1.0 + POISSON_RATIO))
    return 9.0 / 16.0 * np.sqrt(mu_values * gc_values / (3.0 * epsilon))


def stress_values(sig_values: np.ndarray, field: str) -> np.ndarray:
    sig_xx = sig_values[:, 0]
    sig_xy = 0.5 * (sig_values[:, 1] + sig_values[:, 2])
    sig_yy = sig_values[:, 3]

    mean_stress = 0.5 * (sig_xx + sig_yy)
    if field == "sig_vol":
        return mean_stress

    dev_xx = sig_xx - mean_stress
    dev_yy = sig_yy - mean_stress
    return np.sqrt(dev_xx**2 + dev_yy**2 + 2.0 * sig_xy**2)


def load_plot_field(
    xdmf_path: Path,
    field: str,
    target_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if field in {"sig_vol", "sig_dev"}:
        points, cells, sig_values, time = load_field(xdmf_path, "sig", target_time)
        sig_values = sig_values.reshape((-1, 9))
        return points, cells, stress_values(sig_values, field), time

    if field != "sigma_c":
        return load_field(xdmf_path, field, target_time)

    epsilon = split_case_epsilon(xdmf_path)[2]
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"Cannot compute sigma_c without a positive epsilon for {xdmf_path}")

    points, cells, E_values, time = load_field(xdmf_path, "E", target_time)
    _, _, gc_values, _ = load_field(xdmf_path, "gc", target_time)
    return points, cells, sigma_c_values(E_values, gc_values, epsilon), time


def field_limits(xdmf_files: list[Path], field: str, target_time: float | None = None) -> tuple[float, float]:
    if field == "s":
        return 0.0, 1.0

    all_values = []
    for path in xdmf_files:
        _, _, values, _ = load_plot_field(path, field, target_time)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            all_values.append(finite_values)
    if not all_values:
        return 0.0, 1.0

    values = np.concatenate(all_values)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if field == "sig_vol":
        return -500.0, 500.0
    if field == "sig_dev":
        return 0.0, 500.0
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
        return lower - padding, upper + padding
    return lower, upper


def add_case_plot(
    ax: plt.Axes,
    xdmf_path: Path,
    field: str,
    cmap: str,
    vmin: float,
    vmax: float,
    include_a_label: bool = True,
    target_time: float | None = None,
) -> None:
    _, case, epsilon = split_case_epsilon(xdmf_path)
    points, cells, values, time = load_plot_field(xdmf_path, field, target_time)
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)

    if values.size == cells.shape[0]:
        mesh = ax.tripcolor(
            triangulation,
            facecolors=values,
            shading="flat",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    else:
        mesh = ax.tripcolor(
            triangulation,
            values,
            shading="gouraud",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            rasterized=True,
        )
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
    y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
    x_pad = max((x_max - x_min) * 0.015, 1.0e-9)
    y_pad = max((y_max - y_min) * 0.22, 1.0e-9)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    include_time = field in {"s", "sig_vol", "sig_dev"}
    ax.set_title(
        readable_title(case, epsilon, time, include_a_label, include_time=include_time),
        fontsize=CASE_TITLE_SIZE,
    )
    ax.figure._phasefield_mappable = mesh


def write_overview(
    xdmf_files: list[Path],
    output: Path,
    columns: int,
    field: str,
    cmap: str,
    include_corner_label: bool = True,
    include_page_title: bool = True,
    include_a_label: bool = True,
    target_time: float | None = None,
) -> None:
    by_run: dict[tuple[str, float], list[Path]] = {}
    for path in sorted(xdmf_files, key=case_sort_key):
        split, _, epsilon = split_case_epsilon(path)
        by_run.setdefault((split, epsilon), []).append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for (split, epsilon), paths in sorted(by_run.items()):
            vmin, vmax = field_limits(paths, field, target_time)
            paths_by_mode = group_paths_by_mode(paths)
            rows = sum(1 for mode in ("min", "max", "vary") if paths_by_mode.get(mode))
            col_by_key, layout_width_units = build_column_layout(paths_by_mode)
            max_width_units = max(columns, layout_width_units)
            fig_width = 9.36 * max_width_units
            fig_height = 4.25 * rows + 0.9
            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor("white")
            grid = fig.add_gridspec(
                rows,
                max_width_units,
                top=0.94 if not include_page_title else 0.88,
                bottom=0.06,
                hspace=0.28,
                wspace=0.20,
            )
            if include_page_title:
                fig.suptitle(
                    overview_title(field, split, paths, epsilon, include_corner_label),
                    fontsize=PAGE_TITLE_SIZE,
                )
            label = page_corner_label(paths, epsilon)
            if label and include_corner_label:
                fig.text(0.98, 0.98, label, ha="right", va="top", fontsize=22)

            filled_axes = []
            row = 0
            for mode in ("min", "max", "vary"):
                mode_paths = paths_by_mode.get(mode, [])
                if not mode_paths:
                    continue
                beta_values = {beta_from_path(path) for path in paths if math.isfinite(beta_from_path(path))}
                include_beta = len(beta_values) > 1
                for path in mode_paths:
                    span = width_units_for_path(path)
                    col = col_by_key[layout_key_for_path(path, include_beta)]
                    ax = fig.add_subplot(grid[row, col:col + span])
                    add_case_plot(ax, path, field, cmap, vmin, vmax, include_a_label, target_time)
                    filled_axes.append(ax)
                filled_axes[-len(mode_paths)].set_ylabel(mode, fontsize=ROW_LABEL_SIZE)
                row += 1

            mappable = getattr(fig, "_phasefield_mappable", None)
            if mappable is not None:
                colorbar = fig.colorbar(
                    mappable,
                    ax=filled_axes,
                    shrink=0.82,
                    label=field_label(field),
                    pad=0.035,
                )
                colorbar.ax.yaxis.label.set_size(COLORBAR_LABEL_SIZE)
                colorbar.ax.yaxis.labelpad = COLORBAR_LABEL_PAD
                colorbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
            pdf.savefig(fig, dpi=150, bbox_inches="tight", pad_inches=0.18, facecolor="white")
            plt.close(fig)


def group_paths_by_mode(paths: list[Path]) -> dict[str, list[Path]]:
    paths_by_mode: dict[str, list[Path]] = {}
    for path in paths:
        _, case = split_and_case(path)
        mode = case.rsplit("_", 1)[-1]
        paths_by_mode.setdefault(mode, []).append(path)
    for mode_paths in paths_by_mode.values():
        mode_paths.sort(key=case_sort_key)
    return paths_by_mode


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent
    result_roots = args.result_roots or [script_path / "results"]
    output = args.output or script_path / "plots" / f"phasefield_{args.field}_overview.pdf"

    xdmf_files = []
    for root in result_roots:
        xdmf_files.extend(root.rglob("results_*.xdmf"))
    xdmf_files = sorted(set(xdmf_files), key=case_sort_key)
    xdmf_files = filter_xdmf_files(xdmf_files, args.splits, args.a_values)
    if not xdmf_files:
        raise SystemExit("No results_*.xdmf files found.")

    fixed_beta = None
    if not args.include_beta_variants:
        fixed_beta = select_fixed_beta(xdmf_files, args.fixed_beta)
        xdmf_files = filter_by_beta(xdmf_files, fixed_beta)

    fields = args.fields or [args.field]
    for field in fields:
        field_xdmf_files = xdmf_files
        if field in args.collapse_epsilon_fields:
            field_xdmf_files = collapse_epsilon_variants(field_xdmf_files)
        field_output = output
        if args.fields:
            field_output = output.with_name(f"phasefield_{field}_overview{output.suffix}")
        write_overview(
            field_xdmf_files,
            field_output,
            args.columns,
            field,
            args.cmap,
            include_corner_label=not args.hide_corner_label,
            include_page_title=not args.hide_page_title,
            include_a_label=not args.hide_a_label,
            target_time=args.target_time,
        )
        print(f"Wrote {field_output} with {len(field_xdmf_files)} cases.")
    if fixed_beta is not None:
        print(f"Used beta_phi={fixed_beta:g}. Pass --include-beta-variants to plot all beta_phi values.")


if __name__ == "__main__":
    main()
