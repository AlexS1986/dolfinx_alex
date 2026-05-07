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
    r"^results_(?P<case>.+)_(?P<split>spectral|volumetric)(?:_eps(?P<epsilon>[0-9_m]+))?\.xdmf$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the final phase-field s from all results_*.xdmf files below a resource folder."
    )
    parser.add_argument(
        "resource_roots",
        nargs="*",
        type=Path,
        default=[Path("resources")],
        help="Resource folders to scan recursively. Defaults to ./resources.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phasefield_s_overview.pdf"),
        help="Output PDF file.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=8,
        help="Number of plot columns per page.",
    )
    parser.add_argument(
        "--field",
        default="s",
        help="HDF5 function field to plot. Defaults to phase-field s.",
    )
    return parser.parse_args()


def case_sort_key(path: Path) -> tuple[str, int, float, float, float, str]:
    match = RESULT_RE.match(path.name)
    if not match:
        return (path.name, math.inf, math.inf, math.inf, math.inf, "")
    case = match.group("case")
    split = match.group("split")
    mode = case.rsplit("_", 1)[-1]
    mode_order = {"min": 0, "max": 1, "vary": 2}.get(mode, 99)
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")
    return (split, mode_order, beta, a, rho, case)


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


def readable_label(case: str) -> str:
    beta = extract_number(case, r"beta_(0(?:_\d+)?)")
    a = extract_number(case, r"_a_(\d+)")
    rho = extract_number(case, r"_rho_(0_\d+)")
    mode = case.rsplit("_", 1)[-1]
    parts = []
    if math.isfinite(beta):
        parts.append(f"beta={beta:g}")
    if math.isfinite(a):
        parts.append(f"a={a:g}")
    if math.isfinite(rho):
        parts.append(f"rho={rho:g}")
    parts.append(mode)
    return ", ".join(parts)


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


def add_case_plot(ax: plt.Axes, xdmf_path: Path, field: str) -> None:
    split, case = split_and_case(xdmf_path)
    points, cells, values, time = load_final_field(xdmf_path, field)
    triangulation = Triangulation(points[:, 0], points[:, 1], cells)

    mesh = ax.tripcolor(
        triangulation,
        values,
        shading="gouraud",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        rasterized=True,
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{readable_label(case)}\nt={time:.4g}", fontsize=8)
    ax.text(
        0.02,
        0.02,
        split,
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "pad": 1.5, "edgecolor": "none"},
    )
    ax.figure._phasefield_mappable = mesh


def write_overview(xdmf_files: list[Path], output: Path, columns: int, field: str) -> None:
    by_split: dict[str, list[Path]] = {}
    for path in sorted(xdmf_files, key=case_sort_key):
        split, _ = split_and_case(path)
        by_split.setdefault(split, []).append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for split, paths in sorted(by_split.items()):
            paths_by_mode = group_paths_by_mode(paths)
            ordered_paths = [path for mode in ("min", "max", "vary") for path in paths_by_mode.get(mode, [])]
            rows = sum(1 for mode in ("min", "max", "vary") if paths_by_mode.get(mode))
            fig_width = 3.0 * columns
            fig_height = 2.6 * rows + 0.6
            fig, axes = plt.subplots(rows, columns, figsize=(fig_width, fig_height), squeeze=False)
            fig.suptitle(f"Final phase-field {field} overview: {split}", fontsize=14)

            filled_axes = []
            row = 0
            for mode in ("min", "max", "vary"):
                mode_paths = paths_by_mode.get(mode, [])
                if not mode_paths:
                    continue
                for col, path in enumerate(mode_paths):
                    ax = axes[row, col]
                    add_case_plot(ax, path, field)
                    filled_axes.append(ax)
                axes[row, 0].set_ylabel(mode, fontsize=11)
                row += 1

            for ax in axes.ravel():
                if ax not in filled_axes:
                    ax.axis("off")

            mappable = getattr(fig, "_phasefield_mappable", None)
            if mappable is not None:
                fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.82, label=field)
            pdf.savefig(fig, dpi=300)
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
    xdmf_files = []
    for root in args.resource_roots:
        xdmf_files.extend(root.rglob("results_*.xdmf"))
    xdmf_files = sorted(set(xdmf_files), key=case_sort_key)
    if not xdmf_files:
        raise SystemExit("No results_*.xdmf files found.")

    write_overview(xdmf_files, args.output, args.columns, args.field)
    print(f"Wrote {args.output} with {len(xdmf_files)} cases.")


if __name__ == "__main__":
    main()
