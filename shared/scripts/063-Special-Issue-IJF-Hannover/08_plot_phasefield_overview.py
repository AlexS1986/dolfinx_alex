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


RESULT_RE = re.compile(
    r"^results_(?P<case>.+)_(?P<split>spectral|volumetric)(?:_eps(?P<epsilon>[0-9_]+))?\.xdmf$"
)

SIMULATION_FOLDER_RE = re.compile(
    r"^simulation_.*_SPLIT(?P<split>spectral|volumetric)_EPS(?P<epsilon>[0-9_]+)$"
)


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
        "--fixed-beta",
        type=float,
        default=0.01,
        help="Beta value to plot when beta variants are omitted. Defaults to 0.01.",
    )
    parser.add_argument(
        "--include-beta-variants",
        action="store_true",
        help="Plot all beta variants instead of selecting one beta slice.",
    )
    parser.add_argument(
        "--cmap",
        default="coolwarm",
        help="Matplotlib colormap for the contour plot. Defaults to coolwarm, blue to red.",
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


def readable_label(case: str, epsilon: float = math.inf) -> str:
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")
    parts = []
    if math.isfinite(beta):
        parts.append(rf"$\beta={beta:g}$")
    if math.isfinite(a):
        parts.append(rf"$a={a:g}$")
    if math.isfinite(rho):
        parts.append(rf"$\rho={rho:g}$")
    if math.isfinite(epsilon):
        parts.append(rf"$\epsilon={epsilon:g}$")
    return ", ".join(parts)


def readable_title(case: str, epsilon: float, time: float) -> str:
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")

    first_line = []
    second_line = []
    if math.isfinite(a):
        first_line.append(rf"$a={a:g}$")
    if math.isfinite(rho):
        second_line.append(rf"$\rho={rho:g}$")
    second_line.append(rf"$t={time:.3f}$")

    return ", ".join(first_line) + "\n" + ", ".join(second_line)


def page_corner_label(paths: list[Path], epsilon: float) -> str:
    beta_values = sorted({beta_from_path(path) for path in paths if math.isfinite(beta_from_path(path))})
    parts = []
    if beta_values:
        if len(beta_values) == 1:
            parts.append(rf"$\beta={beta_values[0]:g}$")
        else:
            parts.append(rf"$\beta\in[{beta_values[0]:g},{beta_values[-1]:g}]$")
    if math.isfinite(epsilon):
        parts.append(rf"$\epsilon={epsilon:g}$")
    return "\n".join(parts)


def h5_path_from_xdmf(xdmf_path: Path) -> Path:
    return xdmf_path.with_suffix(".h5")


def load_final_field(xdmf_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    h5_path = h5_path_from_xdmf(xdmf_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 companion file: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        points = np.asarray(h5["/Mesh/mesh/geometry"])
        cells = np.asarray(h5["/Mesh/mesh/topology"], dtype=np.int64)
        field_group = h5[f"/Function/{field}"]
        time_key = max(field_group.keys(), key=lambda key: float(key.replace("_", ".")))
        values = np.asarray(field_group[time_key]).reshape(-1)
        time = float(time_key.replace("_", "."))
    return points, cells, values, time


def add_case_plot(ax: plt.Axes, xdmf_path: Path, field: str, cmap: str) -> None:
    _, case, epsilon = split_case_epsilon(xdmf_path)
    points, cells, values, time = load_final_field(xdmf_path, field)
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)

    mesh = ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(readable_title(case, epsilon, time), fontsize=22)
    ax.figure._phasefield_mappable = mesh


def write_overview(xdmf_files: list[Path], output: Path, columns: int, field: str, cmap: str) -> None:
    by_run: dict[tuple[str, float], list[Path]] = {}
    for path in sorted(xdmf_files, key=case_sort_key):
        split, _, epsilon = split_case_epsilon(path)
        by_run.setdefault((split, epsilon), []).append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for (split, epsilon), paths in sorted(by_run.items()):
            paths_by_mode = group_paths_by_mode(paths)
            rows = sum(1 for mode in ("min", "max", "vary") if paths_by_mode.get(mode))
            col_by_key, layout_width_units = build_column_layout(paths_by_mode)
            max_width_units = max(columns, layout_width_units)
            fig_width = 9.36 * max_width_units
            fig_height = 3.72 * rows + 0.5
            fig = plt.figure(figsize=(fig_width, fig_height))
            grid = fig.add_gridspec(
                rows,
                max_width_units,
                top=0.9,
                hspace=0.14,
                wspace=0.20,
            )
            fig.suptitle(f"Final phase-field {field} overview: {split}", fontsize=14)
            label = page_corner_label(paths, epsilon)
            if label:
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
                    add_case_plot(ax, path, field, cmap)
                    filled_axes.append(ax)
                filled_axes[-len(mode_paths)].set_ylabel(mode, fontsize=33)
                row += 1

            mappable = getattr(fig, "_phasefield_mappable", None)
            if mappable is not None:
                colorbar = fig.colorbar(mappable, ax=filled_axes, shrink=0.82, label=field)
                colorbar.ax.yaxis.label.set_size(33)
                colorbar.ax.tick_params(labelsize=33)
            pdf.savefig(fig, dpi=150)
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
    output = args.output or script_path / "plots" / "phasefield_s_overview.pdf"

    xdmf_files = []
    for root in result_roots:
        xdmf_files.extend(root.rglob("results_*.xdmf"))
    xdmf_files = sorted(set(xdmf_files), key=case_sort_key)
    if not xdmf_files:
        raise SystemExit("No results_*.xdmf files found.")

    fixed_beta = None
    if not args.include_beta_variants:
        fixed_beta = select_fixed_beta(xdmf_files, args.fixed_beta)
        xdmf_files = filter_by_beta(xdmf_files, fixed_beta)

    write_overview(xdmf_files, output, args.columns, args.field, args.cmap)
    print(f"Wrote {output} with {len(xdmf_files)} cases.")
    if fixed_beta is not None:
        print(f"Used beta={fixed_beta:g}. Pass --include-beta-variants to plot all beta values.")


if __name__ == "__main__":
    main()
