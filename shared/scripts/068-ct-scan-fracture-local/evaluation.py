#!/usr/bin/env python3
"""Create publication-ready plots from the phase-field graph output.

By default, every run below ``results`` is plotted separately and the figures
are written into the corresponding run directory. Explicit graph-file
arguments can be used to evaluate selected runs. Both graph layouts written
by ``pfmfrac_function.py`` are supported:

    legacy:  time, J_x, J_y, J_z, x_tip, x_tip_prescribed, ...
    current: time, J_x, J_y, J_z, J_x/t_z, ..., x_tip, x_tip_prescribed, ...

The plots use the measured phase-field crack tip, not the prescribed surfing
position. Quantities are nondimensionalized consistently with the plots in
``061-plasticity-fracture-noll-3D`` and exported as PNG, PDF, and PGF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import meshio
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUTPUT_NAMES = (
    "Jx_vs_real_crack_tip.png",
    "Jx_vs_monotonic_real_crack_tip.png",
    "Jx_vs_time.png",
    "Jx_vs_real_crack_tip_unnormalized.png",
    "Jx_vs_monotonic_real_crack_tip_unnormalized.png",
    "Jx_vs_time_unnormalized.png",
)

TIME_COLUMN = 0
LEGACY_COLUMN_COUNT = 12
CURRENT_COLUMN_COUNT = 15
GC_OVERRIDE: float | None = None
LOCAL_DEFAULT_GC = 1.0
WARNED_DEFAULT_GC: set[Path] = set()
CRACK_TIP_HIGHLIGHT_MIN = 10.0
CRACK_TIP_HIGHLIGHT_MAX = 45.0

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": 16,
        "axes.labelsize": 30.6,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "text.usetex": False,
        "pgf.texsystem": "pdflatex",
        "pgf.rcfonts": False,
    }
)


def find_graph_files() -> list[Path]:
    """Return graph files from every run below the results directory."""
    return sorted(RESULTS_DIR.rglob("pfmfrac*_graphs.txt"))


def load_graph_data(graph_file: Path) -> np.ndarray:
    """Load and validate one phase-field graph file."""
    data = np.loadtxt(graph_file, comments="#", ndmin=2)
    if data.shape[1] not in (LEGACY_COLUMN_COUNT, CURRENT_COLUMN_COUNT):
        raise ValueError(
            f"{graph_file} has {data.shape[1]} columns; expected "
            f"{LEGACY_COLUMN_COUNT} (legacy) or {CURRENT_COLUMN_COUNT} (current)."
        )
    return data


def find_mesh_report(graph_file: Path) -> Path | None:
    """Find the mesh report containing the physical specimen bounds."""
    for directory in (graph_file.parent, *graph_file.parents):
        report = directory / "mesh.snap_boundary.txt"
        if report.is_file():
            return report
        if directory == RESULTS_DIR:
            break
    return None


def find_mesh_file(graph_file: Path) -> Path:
    """Find the input mesh copied into the local simulation directory."""
    preferred = graph_file.parent / "dlfx_mesh.xdmf"
    if preferred.is_file():
        return preferred

    candidates = sorted(
        path
        for path in graph_file.parent.glob("*.xdmf")
        if not path.name.startswith("pfmfrac_function")
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(
        f"No mesh.snap_boundary.txt or input mesh XDMF found beside {graph_file}."
    )


def read_specimen_bounds(graph_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read specimen bounds from a report or directly from the XDMF mesh."""
    report = find_mesh_report(graph_file)
    if report is None:
        mesh_file = find_mesh_file(graph_file)
        points = np.asarray(meshio.read(mesh_file).points, dtype=float)
        if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] == 0:
            raise ValueError(f"Invalid three-dimensional points in {mesh_file}.")
        return points[:, :3].min(axis=0), points[:, :3].max(axis=0)

    text = report.read_text(encoding="utf-8")

    def parse_vector(key: str) -> np.ndarray:
        match = re.search(rf"^{key}:\s*\[([^]]+)\]", text, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing {key} in {report}.")
        return np.fromstring(match.group(1), sep=",")

    bounds_min = parse_vector("bounds_min")
    bounds_max = parse_vector("bounds_max")
    if bounds_min.size != 3 or bounds_max.size != 3:
        raise ValueError(f"Invalid three-dimensional bounds in {report}.")
    return bounds_min, bounds_max


def read_gc(graph_file: Path) -> float:
    """Read G_c from CLI override, run metadata, or the directory name."""
    if GC_OVERRIDE is not None:
        return GC_OVERRIDE

    metadata_file = graph_file.parent / "run_parameters.json"
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        try:
            gc = float(metadata["gc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid gc in {metadata_file}.") from error
        if gc <= 0.0:
            raise ValueError(f"G_c must be positive, got {gc} in {metadata_file}.")
        return gc

    match = re.search(r"_Gc([-+0-9.eE]+)_eps", graph_file.parent.name)
    if match is not None:
        gc = float(match.group(1))
        if gc <= 0.0:
            raise ValueError(f"G_c must be positive, got {gc}.")
        return gc

    if graph_file not in WARNED_DEFAULT_GC:
        warnings.warn(
            f"No run_parameters.json found beside {graph_file}; using the "
            f"local default G_c={LOCAL_DEFAULT_GC:g}. Pass --gc to override it.",
            stacklevel=2,
        )
        WARNED_DEFAULT_GC.add(graph_file)
    return LOCAL_DEFAULT_GC


def normalized_plot_data(
    graph_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return dimensionless time, crack position in mm, and normalized J_x."""
    data = load_graph_data(graph_file)
    bounds_min, bounds_max = read_specimen_bounds(graph_file)
    length = bounds_max[1] - bounds_min[1]
    thickness = bounds_max[2] - bounds_min[2]
    if length <= 0.0 or thickness <= 0.0:
        raise ValueError(f"Invalid specimen dimensions for {graph_file}.")

    if data.shape[1] == LEGACY_COLUMN_COUNT:
        jx_per_thickness = data[:, 1] / thickness
        crack_tip = data[:, 4]
        prescribed_tip = data[:, 5]
    else:
        jx_per_thickness = data[:, 4]
        crack_tip = data[:, 7]
        prescribed_tip = data[:, 8]

    time = data[:, TIME_COLUMN]
    if len(time) < 2:
        raise ValueError(f"At least two time values are required in {graph_file}.")
    prescribed_velocity = np.polyfit(time, prescribed_tip, deg=1)[0]
    if not np.isfinite(prescribed_velocity) or prescribed_velocity <= 0.0:
        raise ValueError(f"Invalid prescribed crack velocity in {graph_file}.")

    normalized_time = time / (length / prescribed_velocity)
    normalized_jx = jx_per_thickness / read_gc(graph_file)
    return normalized_time, crack_tip, normalized_jx


def unnormalized_plot_data(
    graph_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return raw time, measured crack position, and integrated 3D J_x."""
    data = load_graph_data(graph_file)
    if data.shape[1] == LEGACY_COLUMN_COUNT:
        crack_tip = data[:, 4]
    else:
        crack_tip = data[:, 7]
    return data[:, TIME_COLUMN], crack_tip, data[:, 1]


def load_jx_and_crack_tip(graph_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite normalized J_x and measured crack-tip values."""
    _, crack_tip, jx = normalized_plot_data(graph_file)

    finite = np.isfinite(jx) & np.isfinite(crack_tip)
    if not np.any(finite):
        raise ValueError(f"{graph_file} contains no finite J_x/crack-tip pairs.")
    return crack_tip[finite], jx[finite]


def load_time_and_jx(graph_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load finite normalized time and J_x values from one graph file."""
    time, _, jx = normalized_plot_data(graph_file)
    finite = np.isfinite(time) & np.isfinite(jx)
    if not np.any(finite):
        raise ValueError(f"{graph_file} contains no finite time/J_x pairs.")
    return time[finite], jx[finite]


def dataset_label(graph_file: Path) -> str:
    """Build a concise label from the simulation directory."""
    try:
        relative_parent = graph_file.parent.relative_to(RESULTS_DIR)
    except ValueError:
        relative_parent = graph_file.parent
    return str(relative_parent)


def zero_based_crack_tip(crack_tip: np.ndarray) -> np.ndarray:
    """Shift a crack-tip history so its first finite position is zero."""
    if crack_tip.size == 0:
        raise ValueError("Cannot normalize an empty crack-tip history.")
    return crack_tip - crack_tip[0]


def highlight_crack_tip_region(ax: plt.Axes) -> None:
    """Highlight the requested crack-tip interval without changing labels."""
    ax.axvspan(
        CRACK_TIP_HIGHLIGHT_MIN,
        CRACK_TIP_HIGHLIGHT_MAX,
        color="tab:orange",
        alpha=0.15,
        zorder=0,
    )


def plot_jx_vs_crack_tip(graph_files: list[Path], output: Path) -> None:
    """Create and save the J_x-versus-real-crack-tip plot."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for graph_file in graph_files:
        crack_tip, jx = load_jx_and_crack_tip(graph_file)
        crack_tip = zero_based_crack_tip(crack_tip)
        ax.plot(
            crack_tip,
            jx,
            marker="o",
            markersize=3,
            linewidth=1,
            label=dataset_label(graph_file),
        )

    highlight_crack_tip_region(ax)
    ax.set_xlabel(r"$x_{\mathrm{tip}}\,[\mathrm{mm}]$")
    ax.set_ylabel(r"$J_x/G_c^{\mathrm{num}}$")
    ax.grid(True, alpha=0.3)
    if len(graph_files) > 1:
        ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_latex_ready_figure(fig, output)
    plt.close(fig)


def plot_jx_vs_monotonic_crack_tip(graph_files: list[Path], output: Path) -> None:
    """Plot J_x against the furthest crack-tip position reached so far."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for graph_file in graph_files:
        crack_tip, jx = load_jx_and_crack_tip(graph_file)
        crack_tip = zero_based_crack_tip(crack_tip)
        monotonic_crack_tip = np.maximum.accumulate(crack_tip)
        ax.plot(
            monotonic_crack_tip,
            jx,
            marker="o",
            markersize=3,
            linewidth=1,
            label=dataset_label(graph_file),
        )

    highlight_crack_tip_region(ax)
    ax.set_xlabel(r"$\max_{\tau\leq t}x_{\mathrm{tip}}(\tau)\,[\mathrm{mm}]$")
    ax.set_ylabel(r"$J_x/G_c^{\mathrm{num}}$")
    ax.grid(True, alpha=0.3)
    if len(graph_files) > 1:
        ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_latex_ready_figure(fig, output)
    plt.close(fig)


def plot_jx_vs_time(graph_files: list[Path], output: Path) -> None:
    """Create and save the J_x-versus-time plot."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for graph_file in graph_files:
        time, jx = load_time_and_jx(graph_file)
        ax.plot(
            time,
            jx,
            marker="o",
            markersize=3,
            linewidth=1,
            label=dataset_label(graph_file),
        )

    ax.set_xlabel(
        r"$t/[(y_{\max}-y_{\min})/\dot{x}_{\mathrm{bc}}]$"
    )
    ax.set_ylabel(r"$J_x/G_c^{\mathrm{num}}$")
    ax.grid(True, alpha=0.3)
    if len(graph_files) > 1:
        ax.legend()

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_latex_ready_figure(fig, output)
    plt.close(fig)


def plot_unnormalized_jx_vs_crack_tip(
    graph_files: list[Path], output: Path, monotonic: bool = False
) -> None:
    """Plot raw integrated J_x against the raw measured crack position."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for graph_file in graph_files:
        _, crack_tip, jx = unnormalized_plot_data(graph_file)
        finite = np.isfinite(crack_tip) & np.isfinite(jx)
        crack_tip = crack_tip[finite]
        jx = jx[finite]
        crack_tip = zero_based_crack_tip(crack_tip)
        if monotonic:
            crack_tip = np.maximum.accumulate(crack_tip)
        ax.plot(
            crack_tip,
            jx,
            marker="o",
            markersize=3,
            linewidth=1,
            label=dataset_label(graph_file),
        )

    highlight_crack_tip_region(ax)
    if monotonic:
        ax.set_xlabel(
            r"$\max_{\tau\leq t}x_{\mathrm{tip}}(\tau)\,[\mathrm{mm}]$"
        )
    else:
        ax.set_xlabel(r"$x_{\mathrm{tip}}\,[\mathrm{mm}]$")
    ax.set_ylabel(r"$J_x$")
    ax.grid(True, alpha=0.3)
    if len(graph_files) > 1:
        ax.legend()

    fig.tight_layout()
    save_latex_ready_figure(fig, output)
    plt.close(fig)


def plot_unnormalized_jx_vs_time(graph_files: list[Path], output: Path) -> None:
    """Plot raw integrated J_x against raw simulation time."""
    fig, ax = plt.subplots(figsize=(10.0, 6.0))

    for graph_file in graph_files:
        time, _, jx = unnormalized_plot_data(graph_file)
        finite = np.isfinite(time) & np.isfinite(jx)
        ax.plot(
            time[finite],
            jx[finite],
            marker="o",
            markersize=3,
            linewidth=1,
            label=dataset_label(graph_file),
        )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$J_x$")
    ax.grid(True, alpha=0.3)
    if len(graph_files) > 1:
        ax.legend()

    fig.tight_layout()
    save_latex_ready_figure(fig, output)
    plt.close(fig)


def save_latex_ready_figure(fig: plt.Figure, output: Path) -> None:
    """Save raster, vector, and directly includable LaTeX versions."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf", ".pgf"):
        target = output.with_suffix(suffix)
        options = {"dpi": 300} if suffix == ".png" else {}
        fig.savefig(target, bbox_inches="tight", **options)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot J_x versus the measured phase-field crack-tip position."
    )
    parser.add_argument(
        "graph_files",
        metavar="GRAPH_FILE",
        nargs="*",
        type=Path,
        help=(
            "graph file(s) to plot; defaults to every pfmfrac*_graphs.txt "
            f"below {RESULTS_DIR}"
        ),
    )
    parser.add_argument(
        "--gc",
        type=float,
        default=None,
        help=(
            "G_c override used for every normalized curve; by default it is "
            "read from run_parameters.json (legacy runs fall back to 1.0)"
        ),
    )
    return parser.parse_args()


def evaluate_run(graph_files: list[Path]) -> None:
    """Create all plot families beside a run's graph files."""
    output_dir = graph_files[0].parent
    outputs = [output_dir / name for name in OUTPUT_NAMES]
    plot_jx_vs_crack_tip(graph_files, outputs[0])
    plot_jx_vs_monotonic_crack_tip(graph_files, outputs[1])
    plot_jx_vs_time(graph_files, outputs[2])
    plot_unnormalized_jx_vs_crack_tip(graph_files, outputs[3])
    plot_unnormalized_jx_vs_crack_tip(graph_files, outputs[4], monotonic=True)
    plot_unnormalized_jx_vs_time(graph_files, outputs[5])
    for base in outputs:
        for suffix in (".png", ".pdf", ".pgf"):
            print(f"Wrote {base.with_suffix(suffix)}")


def main() -> None:
    global GC_OVERRIDE
    args = parse_arguments()
    if args.gc is not None and args.gc <= 0.0:
        raise ValueError(f"--gc must be positive, got {args.gc}.")
    GC_OVERRIDE = args.gc
    graph_files = [path.resolve() for path in args.graph_files] or find_graph_files()
    if not graph_files:
        raise FileNotFoundError(
            f"No pfmfrac*_graphs.txt files found below {RESULTS_DIR}."
        )

    missing = [path for path in graph_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Graph file not found: {missing[0]}")

    graph_files_by_run: dict[Path, list[Path]] = {}
    for graph_file in graph_files:
        graph_files_by_run.setdefault(graph_file.parent, []).append(graph_file)
    for run_graph_files in graph_files_by_run.values():
        evaluate_run(sorted(run_graph_files))


if __name__ == "__main__":
    main()
