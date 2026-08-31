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

# These wide overview canvases are included at \textwidth in the manuscript.
# Use large source text so that the scaled labels are not smaller than \small.
CASE_TITLE_SIZE = 56
PAGE_TITLE_SIZE = 56
COLORBAR_LABEL_SIZE = 56
COLORBAR_TICK_SIZE = 56
COLORBAR_LABEL_PAD = 28
ROW_LABEL_SIZE = 56
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
        default=None,
        help="Optional minimum number of width units per page. By default only occupied columns are rendered.",
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
        "--shared-constant-beta",
        type=float,
        default=None,
        help=(
            "Optional beta_phi dataset supplying shared min/max cases when plotting "
            "a different fixed beta for the varying-porosity cases."
        ),
    )
    parser.add_argument(
        "--include-beta-variants",
        action="store_true",
        help="Plot all beta variants instead of selecting one beta slice.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("min", "max", "vary"),
        default=None,
        help="Optional case modes to include. By default all case modes are plotted.",
    )
    parser.add_argument(
        "--rows-by-rho",
        action="store_true",
        help="Arrange selected cases in rho rows and beta_phi columns for beta comparisons.",
    )
    parser.add_argument(
        "--rows-by-beta",
        action="store_true",
        help="Arrange selected cases in beta_phi rows and rho columns for beta comparisons.",
    )
    parser.add_argument(
        "--show-beta-title",
        action="store_true",
        help="Include beta_phi in each subplot title.",
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
        "--show-corner-label",
        action="store_true",
        help="Print the beta/epsilon annotation in the top right corner. Hidden by default.",
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
        "--inside-title-panels",
        nargs="*",
        default=[],
        help=(
            "Panel letters whose case title should be drawn inside the axes "
            "at the top left instead of above the panel."
        ),
    )
    parser.add_argument(
        "--inside-title-modes",
        nargs="*",
        choices=("min", "max", "vary"),
        default=[],
        help="Case modes whose case title should be drawn at the top left.",
    )
    parser.add_argument(
        "--deformation-scale",
        type=float,
        default=0.0,
        help="Optional displacement scale factor used to plot fields on the deformed mesh.",
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
    include_beta = any(
        len({beta_from_path(path) for path in mode_paths if math.isfinite(beta_from_path(path))}) > 1
        for mode_paths in paths_by_mode.values()
    )
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


def case_mode(path: Path) -> str:
    return split_and_case(path)[1].rsplit("_", 1)[-1]


def select_beta_with_shared_constant_cases(
    xdmf_files: list[Path],
    beta: float,
    shared_constant_beta: float | None,
) -> list[Path]:
    selected = filter_by_beta(xdmf_files, beta)
    if shared_constant_beta is not None and not math.isclose(beta, shared_constant_beta):
        selected.extend(
            path
            for path in xdmf_files
            if math.isclose(beta_from_path(path), shared_constant_beta)
            and case_mode(path) in {"min", "max"}
        )
    return sorted(set(selected), key=case_sort_key)


def filter_xdmf_files(
    xdmf_files: list[Path],
    splits: list[str] | None = None,
    a_values: list[int] | None = None,
    cases: list[str] | None = None,
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
    if cases is not None:
        case_set = set(cases)
        filtered_files = [
            path
            for path in filtered_files
            if case_mode(path) in case_set
        ]
    return filtered_files


def deduplicate_xdmf_files(xdmf_files: list[Path]) -> list[Path]:
    unique_files: dict[tuple[str, str, float], Path] = {}
    preferred_files = sorted(
        xdmf_files,
        key=lambda path: (
            "new_W_whole_boundary" not in str(path),
            case_sort_key(path),
            str(path),
        ),
    )
    for path in preferred_files:
        unique_files.setdefault(split_case_epsilon(path), path)
    return sorted(unique_files.values(), key=case_sort_key)


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
        parts.append(length_label(r"\beta_s", epsilon))
    return ", ".join(parts)


def length_label(symbol: str, value: float) -> str:
    return rf"${symbol}={value:g}\,\mathrm{{mm}}$"


def beta_label(value: float) -> str:
    return rf"$\beta_{{\phi}}={value:g}\,1/\mathrm{{mm}}$"


def mode_label(mode: str) -> str:
    labels = {
        "vary": r"$\mathbf{E}_{\mathrm{var}}$",
        "min": r"$\mathbf{E}_{\mathrm{min}}$",
        "max": r"$\mathbf{E}_{\mathrm{max}}$",
    }
    return labels.get(mode, mode)


def readable_title(
    case: str,
    epsilon: float,
    time: float,
    include_a_label: bool = True,
    include_time: bool = True,
    include_beta_label: bool = False,
) -> str:
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")

    first_line = []
    second_line = []
    if include_beta_label and math.isfinite(beta):
        first_line.append(beta_label(beta))
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
        parts.append(length_label(r"\beta_s", epsilon))
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
    deformation_scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if field in {"sig_vol", "sig_dev"}:
        points, cells, sig_values, time = load_field(xdmf_path, "sig", target_time)
        sig_values = sig_values.reshape((-1, 9))
        return points, cells, stress_values(sig_values, field), time

    if field != "sigma_c":
        points, cells, values, time = load_field(xdmf_path, field, target_time)
        if deformation_scale:
            _, _, displacement, _ = load_field(xdmf_path, "u", time)
            displacement = displacement.reshape((-1, 3))
            points = points.copy()
            points[:, :2] += deformation_scale * displacement[:, :2]
        return points, cells, values, time

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
        return -100.0, 100.0
    if field == "sig_dev":
        return 0.0, 150.0
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 1.0)
        return lower - padding, upper + padding
    return lower, upper


def coordinate_limits(
    xdmf_files: list[Path],
    field: str,
    target_time: float | None = None,
    deformation_scale: float = 0.0,
) -> tuple[float, float, float, float]:
    bounds = []
    for path in xdmf_files:
        points, _, _, _ = load_plot_field(path, field, target_time, deformation_scale)
        bounds.append(
            (
                float(np.min(points[:, 0])),
                float(np.max(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 1])),
            )
        )
    x_min = min(item[0] for item in bounds)
    x_max = max(item[1] for item in bounds)
    y_min = min(item[2] for item in bounds)
    y_max = max(item[3] for item in bounds)
    x_pad = max((x_max - x_min) * 0.015, 1.0e-9)
    y_pad = max((y_max - y_min) * 0.02, 1.0e-9)
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def add_case_plot(
    ax: plt.Axes,
    xdmf_path: Path,
    field: str,
    cmap: str,
    vmin: float,
    vmax: float,
    include_a_label: bool = True,
    target_time: float | None = None,
    include_beta_label: bool = False,
    title_inside: bool = False,
    deformation_scale: float = 0.0,
    plot_bounds: tuple[float, float, float, float] | None = None,
) -> None:
    _, case, epsilon = split_case_epsilon(xdmf_path)
    points, cells, values, time = load_plot_field(xdmf_path, field, target_time, deformation_scale)
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
    if plot_bounds is None:
        x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        x_pad = max((x_max - x_min) * 0.015, 1.0e-9)
        y_pad = max((y_max - y_min) * 0.22, 1.0e-9)
        plot_bounds = x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad
    ax.set_xlim(plot_bounds[0], plot_bounds[1])
    ax.set_ylim(plot_bounds[2], plot_bounds[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    include_time = field in {"s", "sig_vol", "sig_dev"}
    title = readable_title(
        case,
        epsilon,
        time,
        include_a_label,
        include_time=include_time,
        include_beta_label=include_beta_label,
    )
    if title_inside:
        ax.text(
            0.02,
            1.02,
            title,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=CASE_TITLE_SIZE,
            clip_on=False,
        )
    else:
        ax.set_title(title, fontsize=CASE_TITLE_SIZE)
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
    rows_by_rho: bool = False,
    rows_by_beta: bool = False,
    include_beta_title: bool = False,
    inside_title_panels: set[str] | None = None,
    inside_title_modes: set[str] | None = None,
    deformation_scale: float = 0.0,
) -> None:
    inside_title_panels = inside_title_panels or set()
    inside_title_modes = inside_title_modes or set()
    xdmf_files = deduplicate_xdmf_files(xdmf_files)
    by_run: dict[tuple[str, float], list[Path]] = {}
    for path in sorted(xdmf_files, key=case_sort_key):
        split, _, epsilon = split_case_epsilon(path)
        by_run.setdefault((split, epsilon), []).append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for (split, epsilon), paths in sorted(by_run.items()):
            vmin, vmax = field_limits(paths, field, target_time)
            plot_bounds = coordinate_limits(paths, field, target_time, deformation_scale)
            paths_by_mode = group_paths_by_mode(paths)
            if rows_by_rho:
                rho_values = sorted({rho_from_path(path) for path in paths})
                rows = len(rho_values)
                column_keys = sorted({(beta_from_path(path), a_from_path(path)) for path in paths})
                col_by_key = {}
                layout_width_units = 0
                for key in column_keys:
                    representative = next(
                        path
                        for path in paths
                        if (beta_from_path(path), a_from_path(path)) == key
                    )
                    col_by_key[key] = layout_width_units
                    layout_width_units += width_units_for_path(representative)
            elif rows_by_beta:
                beta_values = sorted({beta_from_path(path) for path in paths})
                rows = len(beta_values)
                column_keys = sorted({(rho_from_path(path), a_from_path(path)) for path in paths})
                col_by_key = {}
                layout_width_units = 0
                for key in column_keys:
                    representative = next(
                        path
                        for path in paths
                        if (rho_from_path(path), a_from_path(path)) == key
                    )
                    col_by_key[key] = layout_width_units
                    layout_width_units += width_units_for_path(representative)
            else:
                rows = sum(1 for mode in ("min", "max", "vary") if paths_by_mode.get(mode))
                col_by_key, layout_width_units = build_column_layout(paths_by_mode)
            max_width_units = max(columns or 0, layout_width_units)
            fig_width = 9.36 * max_width_units
            fig_height = 4.25 * rows + 0.9
            fig = plt.figure(figsize=(fig_width, fig_height))
            fig.patch.set_facecolor("white")
            grid = fig.add_gridspec(
                rows,
                max_width_units,
                top=0.94 if not include_page_title else 0.88,
                bottom=0.06,
                hspace=0.16,
                wspace=0.20,
            )
            if include_page_title:
                fig.suptitle("", fontsize=PAGE_TITLE_SIZE)
            label = page_corner_label(paths, epsilon)
            if label and include_corner_label:
                fig.text(0.98, 0.98, label, ha="right", va="top", fontsize=CASE_TITLE_SIZE)

            filled_axes = []
            panel_index = 0
            if rows_by_rho:
                for row, rho in enumerate(rho_values):
                    row_paths = [
                        path for path in paths if math.isclose(rho_from_path(path), rho)
                    ]
                    for path in row_paths:
                        span = width_units_for_path(path)
                        col = col_by_key[(beta_from_path(path), a_from_path(path))]
                        ax = fig.add_subplot(grid[row, col:col + span])
                        add_case_plot(
                            ax,
                            path,
                            field,
                            cmap,
                            vmin,
                            vmax,
                            include_a_label,
                            target_time,
                            include_beta_title,
                            title_inside=(
                                chr(ord("a") + panel_index) in inside_title_panels
                                or case_mode(path) in inside_title_modes
                            ),
                            deformation_scale=deformation_scale,
                            plot_bounds=plot_bounds,
                        )
                        panel_index += 1
                        filled_axes.append(ax)
                    filled_axes[-len(row_paths)].set_ylabel(
                        rf"$\rho={rho:g}$",
                        fontsize=ROW_LABEL_SIZE,
                    )
            elif rows_by_beta:
                for row, beta in enumerate(beta_values):
                    row_paths = [
                        path for path in paths if math.isclose(beta_from_path(path), beta)
                    ]
                    for path in row_paths:
                        span = width_units_for_path(path)
                        col = col_by_key[(rho_from_path(path), a_from_path(path))]
                        ax = fig.add_subplot(grid[row, col:col + span])
                        add_case_plot(
                            ax,
                            path,
                            field,
                            cmap,
                            vmin,
                            vmax,
                            include_a_label,
                            target_time,
                            include_beta_label=False,
                            title_inside=(
                                chr(ord("a") + panel_index) in inside_title_panels
                                or case_mode(path) in inside_title_modes
                            ),
                            deformation_scale=deformation_scale,
                            plot_bounds=plot_bounds,
                        )
                        panel_index += 1
                        filled_axes.append(ax)
                    filled_axes[-len(row_paths)].set_ylabel(
                        rf"${beta:g}$",
                        fontsize=ROW_LABEL_SIZE,
                    )
            else:
                row = 0
                for mode in ("min", "max", "vary"):
                    mode_paths = paths_by_mode.get(mode, [])
                    if not mode_paths:
                        continue
                    include_beta = any(
                        len({beta_from_path(item) for item in candidates if math.isfinite(beta_from_path(item))}) > 1
                        for candidates in paths_by_mode.values()
                    )
                    for path in mode_paths:
                        span = width_units_for_path(path)
                        col = col_by_key[layout_key_for_path(path, include_beta)]
                        ax = fig.add_subplot(grid[row, col:col + span])
                        add_case_plot(
                            ax,
                            path,
                            field,
                            cmap,
                            vmin,
                            vmax,
                            include_a_label,
                            target_time,
                            include_beta_title,
                            title_inside=(
                                chr(ord("a") + panel_index) in inside_title_panels
                                or case_mode(path) in inside_title_modes
                            ),
                            deformation_scale=deformation_scale,
                            plot_bounds=plot_bounds,
                        )
                        panel_index += 1
                        filled_axes.append(ax)
                    row_axes = filled_axes[-len(mode_paths):]
                    row_position = row_axes[0].get_position()
                    fig.text(
                        0.045,
                        0.5 * (row_position.y0 + row_position.y1),
                        mode_label(mode),
                        ha="center",
                        va="center",
                        rotation=90,
                        fontsize=ROW_LABEL_SIZE,
                    )
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
    xdmf_files = filter_xdmf_files(xdmf_files, args.splits, args.a_values, args.cases)
    if not xdmf_files:
        raise SystemExit("No results_*.xdmf files found.")

    fixed_beta = None
    if not args.include_beta_variants:
        fixed_beta = select_fixed_beta(xdmf_files, args.fixed_beta)
        xdmf_files = select_beta_with_shared_constant_cases(
            xdmf_files,
            fixed_beta,
            args.shared_constant_beta,
        )

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
            include_corner_label=args.show_corner_label,
            include_page_title=not args.hide_page_title,
            include_a_label=not args.hide_a_label,
            target_time=args.target_time,
            rows_by_rho=args.rows_by_rho,
            rows_by_beta=args.rows_by_beta,
            include_beta_title=args.show_beta_title,
            inside_title_panels={panel.lower() for panel in args.inside_title_panels},
            inside_title_modes={mode.lower() for mode in args.inside_title_modes},
            deformation_scale=args.deformation_scale,
        )
        print(f"Wrote {field_output} with {len(field_xdmf_files)} cases.")
    if fixed_beta is not None:
        print(f"Used beta_phi={fixed_beta:g}. Pass --include-beta-variants to plot all beta_phi values.")


if __name__ == "__main__":
    main()
