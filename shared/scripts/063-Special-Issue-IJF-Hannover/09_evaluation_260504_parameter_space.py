#!/usr/bin/env python3
"""Evaluate 260504 DCB phase-field results over beta, rho, a, split, and case."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


RESULT_RE = re.compile(
    r"^result_graphs_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)"
    r"_(?P<structure>min|max|var)_(?P<case>min|max|vary)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_m]+))?\.txt$"
)

SIMULATION_FOLDER_RE = re.compile(
    r"^simulation_.*_SPLIT(?P<split>spectral|volumetric)_EPS(?P<epsilon>[0-9_]+)$"
)

QUANTITIES = {
    "Ry": {
        "column": 2,
        "curve_label": r"$R_y$ in N/mm",
        "max_label": r"$\max R_y$ in N/mm",
    },
    "Work": {
        "column": 4,
        "curve_label": r"$W$ in Nmm/mm",
        "max_label": r"$\max W$ in Nmm/mm",
    },
    "Fracture": {
        "column": 5,
        "curve_label": r"$\Pi_\mathrm{frac}$ in Nmm/mm",
        "max_label": r"$\Pi_\mathrm{frac}(R_y=\max R_y)$ in Nmm/mm",
    },
    "Elastic": {
        "column": 6,
        "curve_label": r"$\Pi_\mathrm{el}$ in Nmm/mm",
        "max_label": r"$\Pi_\mathrm{el}(R_y=\max R_y)$ in Nmm/mm",
    },
}

CASE_LINESTYLES = {
    "min": "-",
    "max": (0, (7.0, 3.0)),
    "vary": (0, (4.0, 2.0, 1.4, 2.0)),
}
CASE_MARKERS = {"min": "o", "max": "s", "vary": "^"}
CASE_LABELS = {
    "min": r"$\mathrm{min}$",
    "max": r"$\mathrm{max}$",
    "vary": r"$\mathrm{vary}$",
}
EPSILON_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
RHO_COLORS = {
    0.3: "#c83f49",
    0.6: "#2f6fb7",
}
FALLBACK_RHO_COLORS = [
    "#2f8f6f",
    "#7b5ab6",
    "#c17c2a",
    "#5e6670",
    "#c15a9e",
]
ENERGY_DATASET_COLORS = {
    (0.3, "min"): "#b2182b",
    (0.3, "max"): "#ef8a00",
    (0.3, "vary"): "#d6604d",
    (0.6, "min"): "#2166ac",
    (0.6, "max"): "#67a9cf",
    (0.6, "vary"): "#053061",
}
ENERGY_LINESTYLES = {
    "elastic": "-",
    "fracture": (0, (7.0, 3.0)),
}
WORK_LINESTYLES = {
    "work": "-",
    "total": (0, (2.0, 2.0)),
}
DISSIPATION_COLUMN = 9
WHOLE_BOUNDARY_WORK_COLUMN = 13
DEFAULT_UY_DISPLAY_MAX = 0.019
FAILURE_MARKER_SIZE = 95
FAILURE_MARKER_EDGE_COLOR = "#1f1f1f"
# Plots are included at 0.85\textwidth; maintain at least \small-sized
# labels and annotations after manuscript scaling.
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 22
LEGEND_FONT_SIZE = 22
TITLE_FONT_SIZE = 24
SUPTITLE_FONT_SIZE = 24
SIGMA_AXIS_LABEL_SIZE = 24
SIGMA_TICK_LABEL_SIZE = 22
SIGMA_LEGEND_FONT_SIZE = 22
SIGMA_TITLE_FONT_SIZE = 24
LEGEND_HANDLE_TEXT_PAD = 1.25
LEGEND_LABEL_SPACING = 0.55
AXIS_LABEL_PAD = 9
POISSON_RATIO = 0.3
E_REFERENCE = 210000.0
GC_REFERENCE = 1.0
MU_REFERENCE = E_REFERENCE / (2.0 * (1.0 + POISSON_RATIO))
REFERENCE_LENGTH = 1.0
OMIT_A_IN_TITLES = False
SHOW_PARAMETER_TITLES = False
GC_FIT_A = 1.243657
GC_FIT_B = 3.150239
GC_FIT_C = 2.850765
GC_MIN = 0.1
GC_MAX = 1.0


@dataclass(frozen=True)
class ResultRecord:
    path: Path
    beta: float
    a: int
    rho: float
    case: str
    split: str
    epsilon: float
    max_values: dict[str, float]
    final_values: dict[str, float]
    volume: float | None
    porosity_average: float | None
    E_average: float | None
    Gc_average: float | None
    mu_average: float | None
    sigma_c: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 260504 DCB parameter-space plots from result_graphs files."
    )
    parser.add_argument(
        "result_roots",
        nargs="*",
        type=Path,
        default=None,
        help="Result roots to scan recursively. Defaults to ./results next to this script.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=None,
        help="Folder for generated plots and summary CSV. Defaults to ./plots/260504_evaluation_plots next to this script.",
    )
    parser.add_argument(
        "--x-limit",
        type=float,
        default=None,
        help=(
            "Optional maximum displacement in mm shown in curve plots. "
            f"Defaults to {DEFAULT_UY_DISPLAY_MAX:g} mm."
        ),
    )
    parser.add_argument(
        "--fixed-beta",
        type=float,
        default=0.01,
        help="Beta_phi value used for all plots except vs-beta_phi plots. Defaults to 0.01.",
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
        "--reference-length",
        type=float,
        default=REFERENCE_LENGTH,
        help="Reference length scale in mm used for nondimensional stress axes. Defaults to 1.0.",
    )
    parser.add_argument(
        "--reference-gc",
        type=float,
        default=GC_REFERENCE,
        help="Reference fracture toughness G_c^0 used for nondimensional plot axes. Defaults to 1.0.",
    )
    parser.add_argument(
        "--reference-mu",
        type=float,
        default=MU_REFERENCE,
        help="Reference shear modulus mu^0 used for nondimensional plot axes. Defaults to E^0/(2(1+nu)).",
    )
    parser.add_argument(
        "--omit-a-in-titles",
        action="store_true",
        help="Do not show the a-value in plot titles.",
    )
    parser.add_argument(
        "--show-parameter-titles",
        action="store_true",
        help="Show beta, epsilon and a metadata above plots. Hidden by default because captions normally carry them.",
    )
    return parser.parse_args()


def parse_float(token: str) -> float:
    return float(token.replace("_", ".").replace("m", "-"))


def parse_simulation_folder(path: Path) -> dict[str, object] | None:
    match = SIMULATION_FOLDER_RE.match(path.name)
    if not match:
        return None
    return {
        "split": match.group("split"),
        "epsilon": parse_float(match.group("epsilon")),
    }


def simulation_metadata_for_path(path: Path) -> dict[str, object]:
    for parent in path.parents:
        metadata = parse_simulation_folder(parent)
        if metadata is not None:
            return metadata
    return {}


def parse_result_path(path: Path) -> dict[str, object] | None:
    match = RESULT_RE.match(path.name)
    if not match:
        return None
    folder_metadata = simulation_metadata_for_path(path)
    return {
        "beta": parse_float(match.group("beta")),
        "a": int(match.group("a")),
        "rho": parse_float(match.group("rho")),
        "case": match.group("case"),
        "split": match.group("split"),
        "epsilon": (
            parse_float(match.group("epsilon"))
            if match.group("epsilon")
            else folder_metadata.get("epsilon", 0.015)
        ),
    }


def load_graph_data(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as infile:
        lines = [line for line in infile if line.strip() and not line.startswith("#")]
    data = np.loadtxt(lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def find_volume_metadata(path: Path, case: str, split: str) -> dict[str, float]:
    candidates = sorted(path.parent.glob(f"vol_*_{case}_{split}*.json"))
    if not candidates:
        return {}
    with candidates[0].open("r", encoding="utf-8") as infile:
        return json.load(infile)


def result_xdmf_path_from_graph_path(path: Path) -> Path:
    return path.with_name(path.name.replace("result_graphs_", "results_", 1)).with_suffix(".xdmf")


def load_static_h5_field(xdmf_path: Path, field: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h5_path = xdmf_path.with_suffix(".h5")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5 companion file: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        points = np.asarray(h5["/Mesh/mesh/geometry"])
        cells = np.asarray(h5["/Mesh/mesh/topology"], dtype=np.int64)
        field_group = h5[f"/Function/{field}"]
        time_key = min(field_group.keys(), key=lambda key: float(key.replace("_", ".")))
        values = np.asarray(field_group[time_key]).reshape(-1)
    return points, cells, values


def triangle_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p0 = points[cells[:, 0], :2]
    p1 = points[cells[:, 1], :2]
    p2 = points[cells[:, 2], :2]
    return 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def volume_average_nodal_field(points: np.ndarray, cells: np.ndarray, values: np.ndarray) -> float | None:
    areas = triangle_areas(points, cells)
    if not np.any(areas > 0.0):
        return None
    cell_values = np.mean(values[cells], axis=1)
    finite = np.isfinite(cell_values) & np.isfinite(areas)
    if not np.any(finite):
        return None
    return float(np.sum(cell_values[finite] * areas[finite]) / np.sum(areas[finite]))


def volume_averaged_material_fields(xdmf_path: Path) -> tuple[float | None, float | None]:
    try:
        points, cells, E_values = load_static_h5_field(xdmf_path, "E")
        _, _, gc_values = load_static_h5_field(xdmf_path, "gc")
    except (FileNotFoundError, KeyError, OSError):
        return None, None

    mu_values = E_values / (2.0 * (1.0 + POISSON_RATIO))
    return (
        volume_average_nodal_field(points, cells, gc_values),
        volume_average_nodal_field(points, cells, mu_values),
    )


def effective_gc_from_porosity(porosity_average: float | None) -> float | None:
    if porosity_average is None:
        return None
    phi = max(float(porosity_average), 1.0e-12)
    pore_spacing_ratio = math.sqrt(math.pi / (4.0 * phi)) - 1.0
    gc_value = GC_FIT_A - GC_FIT_B * math.exp(-GC_FIT_C * pore_spacing_ratio)
    return min(max(gc_value, GC_MIN), GC_MAX)


def sigma_c_from_metadata(
    Gc_average: float | None,
    mu_average: float | None,
    epsilon: float,
) -> float | None:
    if Gc_average is None or mu_average is None or epsilon <= 0.0:
        return None
    return 9.0 / 16.0 * math.sqrt(mu_average * Gc_average / (3.0 * epsilon))


def collect_records(resource_roots: list[Path]) -> list[ResultRecord]:
    records = []
    for root in resource_roots:
        for path in root.rglob("result_graphs_*.txt"):
            parsed = parse_result_path(path)
            if parsed is None:
                continue
            data = load_graph_data(path)
            failure_index = int(np.argmax(np.abs(data[:, QUANTITIES["Ry"]["column"]])))
            max_values = {
                name: float(np.max(np.abs(data[:, spec["column"]])))
                for name, spec in QUANTITIES.items()
            }
            for name in ("Fracture", "Elastic"):
                max_values[name] = float(
                    np.abs(data[failure_index, QUANTITIES[name]["column"]])
                )
            final_values = {
                name: float(np.abs(data[-1, spec["column"]]))
                for name, spec in QUANTITIES.items()
            }
            volume_metadata = find_volume_metadata(path, parsed["case"], parsed["split"])
            E_average = volume_metadata.get("E_average")
            porosity_average = volume_metadata.get("porosity_average")
            Gc_average, mu_average = volume_averaged_material_fields(result_xdmf_path_from_graph_path(path))
            if Gc_average is None:
                Gc_average = effective_gc_from_porosity(porosity_average)
            if mu_average is None and E_average is not None:
                mu_average = float(E_average) / (2.0 * (1.0 + POISSON_RATIO))
            records.append(
                ResultRecord(
                    path=path,
                    beta=parsed["beta"],
                    a=parsed["a"],
                    rho=parsed["rho"],
                    case=parsed["case"],
                    split=parsed["split"],
                    epsilon=parsed["epsilon"],
                    max_values=max_values,
                    final_values=final_values,
                    volume=volume_metadata.get("vol"),
                    porosity_average=porosity_average,
                    E_average=E_average,
                    Gc_average=Gc_average,
                    mu_average=mu_average,
                    sigma_c=sigma_c_from_metadata(Gc_average, mu_average, parsed["epsilon"]),
                )
            )
    return sorted(records, key=record_sort_key)


def record_sort_key(record: ResultRecord) -> tuple[str, str, int, float, float, float]:
    return (record.split, record.case, record.a, record.beta, record.rho, record.epsilon)


def label_for_record(record: ResultRecord, include_beta: bool = True) -> str:
    parts = []
    if include_beta:
        parts.append(beta_label(record.beta))
    parts.extend([
        length_label("a", record.a),
        rf"$\rho={record.rho:g}$",
        length_label(r"\epsilon", record.epsilon),
    ])
    return ", ".join(parts)


def fixed_beta_label(fixed_beta: float | None) -> str:
    if fixed_beta is None:
        return ""
    return ", " + beta_label(fixed_beta)


def title_parameter_label(
    a_value: float,
    fixed_beta: float | None,
    epsilon: float | None = None,
) -> str:
    if not SHOW_PARAMETER_TITLES:
        return ""
    parts = []
    if not OMIT_A_IN_TITLES:
        parts.append(length_label("a", a_value))
    if epsilon is not None:
        parts.append(length_label(r"\epsilon", epsilon))

    label = ", ".join(parts)
    beta = fixed_beta_label(fixed_beta)
    if label:
        return label + beta
    return beta.removeprefix(", ")


def case_label(case: str) -> str:
    return CASE_LABELS.get(case, rf"$\mathrm{{{case}}}$")


def quantity_label(quantity: str, kind: str = "max_label") -> str:
    return QUANTITIES[quantity][kind]


def displacement_scale() -> float:
    return math.sqrt(GC_REFERENCE * REFERENCE_LENGTH / (2.0 * MU_REFERENCE))


def stress_scale() -> float:
    return math.sqrt(2.0 * MU_REFERENCE * GC_REFERENCE / REFERENCE_LENGTH)


def force_scale() -> float:
    return stress_scale() * REFERENCE_LENGTH


def normalized_metric_value(metric: str, value: float) -> float:
    return value


def peak_reaction_index(data: np.ndarray) -> int:
    return int(np.argmax(np.abs(data[:, QUANTITIES["Ry"]["column"]])))


def work_at_peak_reaction(data: np.ndarray) -> float:
    return float(
        np.abs(data[peak_reaction_index(data), QUANTITIES["Work"]["column"]])
    )


def normalized_metric_values(metric: str, values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def length_label(symbol: str, value: float) -> str:
    return rf"${symbol}={value:g}\,\mathrm{{mm}}$"


def length_axis_label(symbol: str) -> str:
    return rf"${symbol}\,\mathrm{{[mm]}}$"


def beta_label(value: float) -> str:
    return rf"$\beta_{{\phi}}={value:g}\,1/\mathrm{{mm}}$"


def beta_axis_label() -> str:
    return r"$\beta_{\phi}$ in $1/\mathrm{mm}$"


def normalized_epsilon(epsilon: float) -> float:
    return epsilon / REFERENCE_LENGTH


def normalized_beta(beta: float) -> float:
    return beta / REFERENCE_LENGTH


def normalized_a(a_value: float) -> float:
    return a_value / REFERENCE_LENGTH


def float_filename_token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "_")


def nice_axis_upper_limit(max_value: float) -> float:
    if max_value <= 0.0 or not math.isfinite(max_value):
        return 1.0

    padded_value = max_value * 1.03
    magnitude = 10.0 ** math.floor(math.log10(padded_value))
    normalized = padded_value / magnitude
    for candidate in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0):
        if normalized <= candidate:
            return candidate * magnitude
    return 10.0 * magnitude


def color_for_rho(rho: float) -> str:
    for known_rho, color in RHO_COLORS.items():
        if math.isclose(rho, known_rho):
            return color
    index = int(abs(round(rho * 1000))) % len(FALLBACK_RHO_COLORS)
    return FALLBACK_RHO_COLORS[index]


def energy_color_for_record(record: ResultRecord) -> str:
    for (rho, case), color in ENERGY_DATASET_COLORS.items():
        if math.isclose(record.rho, rho) and record.case == case:
            return color
    return color_for_rho(record.rho)


def marker_for_epsilon(epsilon: float, epsilon_values: list[float]) -> str:
    for index, known_epsilon in enumerate(epsilon_values):
        if math.isclose(epsilon, known_epsilon):
            return EPSILON_MARKERS[index % len(EPSILON_MARKERS)]
    return "o"


def line_style_for_case(case: str) -> str:
    return CASE_LINESTYLES.get(case, "-")


def marker_for_case(case: str) -> str:
    return CASE_MARKERS.get(case, "o")


def split_output_folder(output_folder: Path, split: str) -> Path:
    split_folder = output_folder / split
    split_folder.mkdir(parents=True, exist_ok=True)
    return split_folder


def remove_existing_plots(output_folder: Path, pattern: str) -> None:
    for folder in (output_folder, output_folder / "spectral", output_folder / "volumetric"):
        if not folder.exists():
            continue
        for plot_path in folder.glob(pattern):
            plot_path.unlink()


def style_legend(legend, font_size: int = LEGEND_FONT_SIZE) -> None:
    if legend is None:
        return
    legend.set_frame_on(True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.86)
    frame.set_edgecolor("0.82")
    frame.set_linewidth(0.7)
    legend.set_title(legend.get_title().get_text(), prop={"size": font_size})
    for text in legend.get_texts():
        text.set_fontsize(font_size)
    legend.handletextpad = LEGEND_HANDLE_TEXT_PAD
    legend.labelspacing = LEGEND_LABEL_SPACING
    try:
        handles = legend.legend_handles
    except AttributeError:
        handles = legend.legendHandles
    for handle in handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(2.4)


def select_fixed_beta(records: list[ResultRecord], requested_beta: float | None) -> float:
    beta_values = sorted({record.beta for record in records})
    if not beta_values:
        raise SystemExit("No beta values found in result records.")
    if requested_beta is None:
        return beta_values[0]
    for beta in beta_values:
        if math.isclose(beta, requested_beta):
            return beta
    available = ", ".join(f"{beta:g}" for beta in beta_values)
    raise SystemExit(f"Requested --fixed-beta {requested_beta:g} was not found. Available beta values: {available}")


def filter_by_beta(records: list[ResultRecord], beta: float) -> list[ResultRecord]:
    return [record for record in records if math.isclose(record.beta, beta)]


def filter_records(
    records: list[ResultRecord],
    splits: list[str] | None = None,
    a_values: list[int] | None = None,
) -> list[ResultRecord]:
    filtered_records = records
    if splits is not None:
        split_set = set(splits)
        filtered_records = [record for record in filtered_records if record.split in split_set]
    if a_values is not None:
        a_set = set(a_values)
        filtered_records = [record for record in filtered_records if record.a in a_set]
    return filtered_records


def case_order(case: str) -> int:
    return {"min": 0, "max": 1, "vary": 2}.get(case, 99)


def format_axes(
    ax: plt.Axes,
    title_size: int = TITLE_FONT_SIZE,
    axis_label_size: int = AXIS_LABEL_SIZE,
    tick_label_size: int = TICK_LABEL_SIZE,
) -> None:
    ax.xaxis.label.set_size(axis_label_size)
    ax.yaxis.label.set_size(axis_label_size)
    ax.xaxis.labelpad = AXIS_LABEL_PAD
    ax.yaxis.labelpad = AXIS_LABEL_PAD
    ax.title.set_size(title_size)
    ax.tick_params(axis="both", labelsize=tick_label_size)


def format_suptitle(fig: plt.Figure) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_size(SUPTITLE_FONT_SIZE)


def summary_figure_size(axis_count: int) -> tuple[float, float]:
    if axis_count == 1:
        return (10.0, 6.2)
    return (6.3 * axis_count, 5.2)


def write_summary_csv(records: list[ResultRecord], output_folder: Path) -> None:
    path = output_folder / "summary_260504.csv"
    fieldnames = [
        "split",
        "case",
        "beta",
        "a",
        "rho",
        "epsilon",
        "volume",
        "porosity_average",
        "porosity_average_times_volume",
        "E_average",
        "Gc_average",
        "mu_average",
        "sigma_c",
    ]
    for prefix in ("max", "final"):
        fieldnames.extend(f"{prefix}_{quantity}" for quantity in QUANTITIES)
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "split": record.split,
                "case": record.case,
                "beta": record.beta,
                "a": record.a,
                "rho": record.rho,
                "epsilon": record.epsilon,
                "volume": "" if record.volume is None else record.volume,
                "porosity_average": "" if record.porosity_average is None else record.porosity_average,
                "porosity_average_times_volume": (
                    ""
                    if record.volume is None or record.porosity_average is None
                    else record.volume * record.porosity_average
                ),
                "E_average": "" if record.E_average is None else record.E_average,
                "Gc_average": "" if record.Gc_average is None else record.Gc_average,
                "mu_average": "" if record.mu_average is None else record.mu_average,
                "sigma_c": "" if record.sigma_c is None else record.sigma_c,
            }
            row.update({f"max_{name}": value for name, value in record.max_values.items()})
            row.update({f"final_{name}": value for name, value in record.final_values.items()})
            writer.writerow(row)


def uy_display_maximum(x_limit: float | None) -> float:
    return DEFAULT_UY_DISPLAY_MAX if x_limit is None else x_limit


def whole_boundary_work_values(data: np.ndarray, path: Path) -> np.ndarray:
    if data.shape[1] <= WHOLE_BOUNDARY_WORK_COLUMN:
        raise ValueError(
            f"{path} does not contain W_sigma_trap_boundary. "
            "Regenerate the simulation results with the updated "
            "000_template/01_phasefield_dcb_260504_folder.py before "
            "creating Fig. 17 panels (b) and (d)."
        )
    return np.abs(data[:, WHOLE_BOUNDARY_WORK_COLUMN])


def plot_curves(records: list[ResultRecord], output_folder: Path, x_limit: float | None, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, "*_vs_uy_*.png")

    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for a_value, epsilon in sorted({(record.a, record.epsilon) for record in split_records}):
            parameter_records = sorted(
                [
                    record
                    for record in split_records
                    if record.a == a_value and math.isclose(record.epsilon, epsilon)
                ],
                key=lambda record: (record.rho, case_order(record.case)),
            )
            for quantity, spec in QUANTITIES.items():
                fig, ax = plt.subplots(figsize=(10.0, 6.2))
                max_displacement = 0.0
                for record in parameter_records:
                    data = load_graph_data(record.path)
                    displacement = np.abs(data[:, 1])
                    values = normalized_metric_values(quantity, np.abs(data[:, spec["column"]]))
                    failure_index = peak_reaction_index(data)
                    max_displacement = max(max_displacement, float(np.max(displacement)))
                    label = rf"$\rho={record.rho:g}$, {record.case}"
                    ax.plot(
                        displacement,
                        values,
                        label=label,
                        color=color_for_rho(record.rho),
                        linewidth=1.6,
                        linestyle=line_style_for_case(record.case),
                    )
                    ax.scatter(
                        displacement[failure_index],
                        values[failure_index],
                        s=FAILURE_MARKER_SIZE,
                        marker="X",
                        color=color_for_rho(record.rho),
                        edgecolor=FAILURE_MARKER_EDGE_COLOR,
                        linewidth=0.85,
                        zorder=5,
                    )
                ax.set_title(title_parameter_label(a_value, fixed_beta, epsilon))
                ax.set_xlabel(r"$u_y$ in mm")
                ax.set_ylabel(quantity_label(quantity, "curve_label"))
                ax.set_xlim(0.0, uy_display_maximum(x_limit))
                ax.grid(True, alpha=0.3)
                style_legend(ax.legend(fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False, handlelength=3.8, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING))
                format_axes(ax)
                fig.tight_layout()
                fig.savefig(
                    split_output_folder(output_folder, split)
                    / (
                        f"{quantity}_vs_uy_{split}_a_{a_value}"
                        f"_eps{float_filename_token(epsilon)}.png"
                    ),
                    dpi=300,
                )
                plt.close(fig)
            plot_energy_components_curve(
                parameter_records,
                output_folder,
                split,
                a_value,
                epsilon,
                x_limit,
                fixed_beta,
            )
            plot_response_energy_grid(
                parameter_records,
                output_folder,
                split,
                a_value,
                epsilon,
                x_limit,
            )


def plot_work_balance_curve(
    parameter_records: list[ResultRecord],
    output_folder: Path,
    split: str,
    a_value: int,
    epsilon: float,
    x_limit: float | None,
    fixed_beta: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    max_displacement = 0.0
    max_value = 0.0

    for record in parameter_records:
        data = load_graph_data(record.path)
        displacement = np.abs(data[:, 1])
        work = np.abs(data[:, QUANTITIES["Work"]["column"]])
        elastic = np.abs(data[:, QUANTITIES["Elastic"]["column"]])
        fracture = np.abs(data[:, QUANTITIES["Fracture"]["column"]])
        total_energy = elastic + fracture
        color = energy_color_for_record(record)
        max_displacement = max(max_displacement, float(np.max(displacement)))
        max_value = max(max_value, float(np.max(work)), float(np.max(total_energy)))

        ax.plot(
            displacement,
            work,
            color=color,
            linestyle=WORK_LINESTYLES["work"],
            linewidth=1.8,
        )
        ax.plot(
            displacement,
            total_energy,
            color=color,
            linestyle=WORK_LINESTYLES["total"],
            linewidth=1.8,
        )

    ax.set_title(title_parameter_label(a_value, fixed_beta, epsilon))
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(r"$W,\ \Pi_\mathrm{el}+\Pi_\mathrm{frac}$ in Nmm/mm")
    ax.set_xlim(0.0, uy_display_maximum(x_limit))
    ax.set_ylim(0.0, nice_axis_upper_limit(max_value))
    ax.grid(True, alpha=0.3)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=energy_color_for_record(record),
            linewidth=2.4,
            label=rf"$\rho={record.rho:g}$, {record.case}",
        )
        for record in parameter_records
    ]
    quantity_handles = [
        Line2D(
            [0],
            [0],
            color="#2f2f2f",
            linestyle=WORK_LINESTYLES[name],
            linewidth=2.2,
            label=label,
        )
        for name, label in (
            ("work", r"$W$"),
            ("total", r"$\Pi_\mathrm{el}+\Pi_\mathrm{frac}$"),
        )
    ]
    first_legend = ax.legend(
        handles=dataset_handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        loc="upper left",
        handlelength=3.2,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    ax.add_artist(first_legend)
    style_legend(first_legend)
    second_legend = ax.legend(
        handles=quantity_handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper right",
        handlelength=3.4,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    style_legend(second_legend)

    format_axes(ax)
    fig.tight_layout()
    fig.savefig(
        split_output_folder(output_folder, split)
        / (
            f"Work_vs_uy_{split}_a_{a_value}"
            f"_eps{float_filename_token(epsilon)}.png"
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_energy_components_curve(
    parameter_records: list[ResultRecord],
    output_folder: Path,
    split: str,
    a_value: int,
    epsilon: float,
    x_limit: float | None,
    fixed_beta: float,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    max_displacement = 0.0
    max_energy = 0.0

    for record in parameter_records:
        data = load_graph_data(record.path)
        displacement = np.abs(data[:, 1])
        elastic = np.abs(data[:, QUANTITIES["Elastic"]["column"]])
        fracture = np.abs(data[:, QUANTITIES["Fracture"]["column"]])
        color = energy_color_for_record(record)
        max_displacement = max(max_displacement, float(np.max(displacement)))
        max_energy = max(max_energy, float(np.max(elastic)), float(np.max(fracture)))

        for name, values in (
            ("elastic", elastic),
            ("fracture", fracture),
        ):
            ax.plot(
                displacement,
                values,
                color=color,
                linestyle=ENERGY_LINESTYLES[name],
                linewidth=1.8,
            )

    ax.set_title(title_parameter_label(a_value, fixed_beta, epsilon))
    ax.set_xlabel(r"$u_y$ in mm")
    ax.set_ylabel(r"$\Pi$ in Nmm/mm")
    ax.set_xlim(0.0, uy_display_maximum(x_limit))
    ax.set_ylim(0.0, nice_axis_upper_limit(max_energy))
    ax.grid(True, alpha=0.3)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=energy_color_for_record(record),
            linewidth=2.4,
            label=rf"$\rho={record.rho:g}$, {record.case}",
        )
        for record in parameter_records
    ]
    energy_handles = [
        Line2D(
            [0],
            [0],
            color="#2f2f2f",
            linestyle=ENERGY_LINESTYLES[name],
            linewidth=2.2,
            label=label,
        )
        for name, label in (
            ("elastic", r"$\Pi_\mathrm{el}$"),
            ("fracture", r"$\Pi_\mathrm{frac}$"),
        )
    ]
    first_legend = ax.legend(
        handles=dataset_handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        ncol=2,
        loc="upper left",
        handlelength=3.2,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    ax.add_artist(first_legend)
    style_legend(first_legend)
    second_legend = ax.legend(
        handles=energy_handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper right",
        handlelength=3.4,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    style_legend(second_legend)

    format_axes(ax)
    fig.tight_layout()
    fig.savefig(
        split_output_folder(output_folder, split)
        / (
            f"Energy_components_vs_uy_{split}_a_{a_value}"
            f"_eps{float_filename_token(epsilon)}.png"
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_response_energy_grid(
    parameter_records: list[ResultRecord],
    output_folder: Path,
    split: str,
    a_value: int,
    epsilon: float,
    x_limit: float | None,
) -> None:
    """Plot the force response and energy balance in a manuscript four-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 12.5), sharex=True)
    (ax_force, ax_work), (ax_components, ax_balance) = axes
    max_displacement = 0.0
    max_force = 0.0
    max_work = 0.0
    max_components = 0.0
    max_balance = 0.0

    for record in parameter_records:
        data = load_graph_data(record.path)
        displacement = np.abs(data[:, 1])
        reaction = np.abs(data[:, QUANTITIES["Ry"]["column"]])
        work = whole_boundary_work_values(data, record.path)
        fracture = np.abs(data[:, QUANTITIES["Fracture"]["column"]])
        elastic = np.abs(data[:, QUANTITIES["Elastic"]["column"]])
        dissipation = np.abs(data[:, DISSIPATION_COLUMN])
        total_energy = elastic + fracture + dissipation
        peak_index = peak_reaction_index(data)
        color = energy_color_for_record(record)

        max_displacement = max(max_displacement, float(np.max(displacement)))
        max_force = max(max_force, float(np.max(reaction)))
        max_work = max(max_work, float(np.max(work)))
        max_components = max(max_components, float(np.max(elastic)), float(np.max(fracture)))
        max_balance = max(max_balance, float(np.max(work)), float(np.max(total_energy)))

        ax_force.plot(displacement, reaction, color=color, linewidth=1.8)
        ax_work.plot(displacement, work, color=color, linewidth=1.8)
        ax_components.plot(
            displacement,
            elastic,
            color=color,
            linestyle=ENERGY_LINESTYLES["elastic"],
            linewidth=1.8,
        )
        ax_components.plot(
            displacement,
            fracture,
            color=color,
            linestyle=ENERGY_LINESTYLES["fracture"],
            linewidth=1.8,
        )
        ax_balance.plot(
            displacement,
            work,
            color=color,
            linestyle=WORK_LINESTYLES["work"],
            linewidth=1.8,
        )
        ax_balance.plot(
            displacement,
            total_energy,
            color=color,
            linestyle=WORK_LINESTYLES["total"],
            linewidth=1.8,
        )

        for ax, values in (
            (ax_force, reaction),
            (ax_work, work),
            (ax_components, elastic),
            (ax_components, fracture),
            (ax_balance, work),
            (ax_balance, total_energy),
        ):
            ax.scatter(
                displacement[peak_index],
                values[peak_index],
                s=FAILURE_MARKER_SIZE,
                marker="X",
                color=color,
                edgecolor=FAILURE_MARKER_EDGE_COLOR,
                linewidth=0.85,
                zorder=5,
            )

    x_upper = uy_display_maximum(x_limit)
    for panel_label, ax in zip(("a", "b", "c", "d"), axes.ravel()):
        ax.text(
            0.02,
            0.95,
            rf"\textbf{{({panel_label})}}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TITLE_FONT_SIZE,
        )
        ax.set_xlim(0.0, x_upper)
        ax.grid(True, alpha=0.3)
        format_axes(ax)

    ax_force.set_ylabel(r"$R_y$ in N/mm")
    ax_work.set_ylabel(r"$W_{\partial\Omega}$ in Nmm/mm")
    ax_components.set_ylabel(r"$\Pi$ in Nmm/mm")
    ax_balance.set_ylabel(r"$W_{\partial\Omega},\ \Pi_\mathrm{tot}$ in Nmm/mm")
    ax_components.set_xlabel(r"$u_y$ in mm")
    ax_balance.set_xlabel(r"$u_y$ in mm")
    ax_force.set_ylim(0.0, nice_axis_upper_limit(max_force))
    ax_work.set_ylim(0.0, nice_axis_upper_limit(max_work))
    ax_components.set_ylim(0.0, nice_axis_upper_limit(max_components))
    ax_balance.set_ylim(0.0, nice_axis_upper_limit(max_balance))

    dataset_handles = [
        Line2D(
            [0],
            [0],
            color=energy_color_for_record(record),
            linewidth=2.4,
            label=rf"$\rho={record.rho:g}$, {case_label(record.case)}",
        )
        for record in parameter_records
    ]
    curve_handles = [
        Line2D([0], [0], color="#2f2f2f", linewidth=2.2, label=r"$\Pi_\mathrm{el}$ / $W_{\partial\Omega}$"),
        Line2D(
            [0],
            [0],
            color="#2f2f2f",
            linestyle=ENERGY_LINESTYLES["fracture"],
            linewidth=2.2,
            label=r"$\Pi_\mathrm{frac}$ / $\Pi_\mathrm{tot}$",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="#2f2f2f",
            linewidth=0.0,
            markersize=9,
            label=r"$R_y=\max R_y$",
        ),
    ]
    dataset_legend = fig.legend(
        handles=dataset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.8,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    style_legend(dataset_legend)
    curve_legend = fig.legend(
        handles=curve_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=3,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=2.8,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )
    style_legend(curve_legend)

    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.10, right=0.98, hspace=0.20, wspace=0.20)
    fig.savefig(
        split_output_folder(output_folder, split)
        / (
            f"Response_energy_grid_vs_uy_{split}_a_{a_value}"
            f"_eps{float_filename_token(epsilon)}.png"
        ),
        dpi=300,
    )
    plt.close(fig)


def plot_metric_vs_beta(records: list[ResultRecord], output_folder: Path, metric: str) -> None:
    remove_existing_plots(output_folder, f"max_{metric}_vs_beta_*.png")

    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            if not case_records:
                continue

            a_values = sorted({record.a for record in case_records})
            fig, axes = plt.subplots(1, len(a_values), figsize=(6.3 * len(a_values), 5.2), squeeze=False)
            for ax, a_value in zip(axes.ravel(), a_values):
                subset = [record for record in case_records if record.a == a_value]
                epsilon_values = sorted({record.epsilon for record in subset})
                for rho, epsilon in sorted({(record.rho, record.epsilon) for record in subset}):
                    rho_subset = sorted(
                        [
                            record
                            for record in subset
                            if record.rho == rho and math.isclose(record.epsilon, epsilon)
                        ],
                        key=lambda record: record.beta,
                    )
                    label = rf"$\rho={rho:g}$, " + length_label(r"\epsilon", epsilon)
                    ax.plot(
                        [record.beta for record in rho_subset],
                        [normalized_metric_value(metric, record.max_values[metric]) for record in rho_subset],
                        color=color_for_rho(rho),
                        linestyle=line_style_for_case(case),
                        marker=marker_for_epsilon(epsilon, epsilon_values),
                        linewidth=1.8,
                        label=label,
                    )
                ax.set_xscale("log")
                ax.set_title(length_label("a", a_value))
                ax.set_xlabel(beta_axis_label())
                ax.set_ylabel(quantity_label(metric))
                ax.grid(True, alpha=0.3)
                style_legend(ax.legend(frameon=False, handlelength=3.8, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
                format_axes(ax)
            fig.tight_layout()
            fig.savefig(split_output_folder(output_folder, split) / f"max_{metric}_vs_beta_{split}_{case}.png", dpi=300)
            plt.close(fig)


def plot_metric_vs_rho(records: list[ResultRecord], output_folder: Path, metric: str, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, f"max_{metric}_vs_rho_*.png")

    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            if not case_records:
                continue

            a_values = sorted({record.a for record in case_records})
            fig, axes = plt.subplots(1, len(a_values), figsize=(6.3 * len(a_values), 5.2), squeeze=False)
            for ax, a_value in zip(axes.ravel(), a_values):
                subset = [record for record in case_records if record.a == a_value]
                rho_values = sorted({record.rho for record in subset})
                for epsilon in sorted({record.epsilon for record in subset}):
                    epsilon_subset = sorted(
                        [
                            record
                            for record in subset
                            if math.isclose(record.epsilon, epsilon)
                        ],
                        key=lambda record: record.rho,
                    )
                    label = length_label(r"\epsilon", epsilon)
                    x_values = [record.rho for record in epsilon_subset]
                    y_values = [normalized_metric_value(metric, record.max_values[metric]) for record in epsilon_subset]
                    ax.plot(
                        x_values,
                        y_values,
                        color="#7f8790",
                        linestyle=line_style_for_case(case),
                        linewidth=1.8,
                        label=label,
                    )
                    for rho, value in zip(x_values, y_values):
                        ax.scatter(
                            rho,
                            value,
                            color=color_for_rho(rho),
                            marker=marker_for_epsilon(epsilon, sorted({record.epsilon for record in subset})),
                            s=45,
                            zorder=3,
                        )
                ax.set_xlabel(r"$\rho$")
                ax.set_title(title_parameter_label(a_value, fixed_beta))
                ax.set_xticks(rho_values)
                ax.set_ylabel(quantity_label(metric))
                ax.grid(True, alpha=0.3)
                style_legend(ax.legend(frameon=False, handlelength=3.8, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
                format_axes(ax)
            fig.tight_layout()
            fig.savefig(split_output_folder(output_folder, split) / f"max_{metric}_vs_rho_{split}_{case}.png", dpi=300)
            plt.close(fig)


def plot_metric_vs_epsilon(records: list[ResultRecord], output_folder: Path, metric: str, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, f"max_{metric}_vs_epsilon_*.png")

    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue

        a_values = sorted({record.a for record in split_records})
        fig, axes = plt.subplots(
            1,
            len(a_values),
            figsize=summary_figure_size(len(a_values)),
            squeeze=False,
        )
        for ax, a_value in zip(axes.ravel(), a_values):
            subset = [record for record in split_records if record.a == a_value]
            for rho in sorted({record.rho for record in subset}):
                rho_records = [record for record in subset if math.isclose(record.rho, rho)]
                for case in sorted({record.case for record in rho_records}, key=case_order):
                    case_subset = [record for record in rho_records if record.case == case]
                    rho_subset = sorted(
                        [
                            record
                            for record in case_subset
                        ],
                        key=lambda record: record.epsilon,
                    )
                    if not rho_subset:
                        continue

                    label = rf"$\rho={rho:g}$, {case_label(case)}"
                    ax.plot(
                        [record.epsilon for record in rho_subset],
                        [normalized_metric_value(metric, record.max_values[metric]) for record in rho_subset],
                        color=color_for_rho(rho),
                        linestyle=line_style_for_case(case),
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )
            ax.set_title(title_parameter_label(a_value, fixed_beta))
            ax.set_xlabel(length_axis_label(r"\epsilon"))
            ax.set_ylabel(quantity_label(metric))
            ax.grid(True, alpha=0.3)
            style_legend(
                ax.legend(
                    frameon=False,
                    handlelength=3.8,
                    handletextpad=LEGEND_HANDLE_TEXT_PAD,
                    labelspacing=LEGEND_LABEL_SPACING,
                    fontsize=SIGMA_LEGEND_FONT_SIZE,
                ),
                font_size=SIGMA_LEGEND_FONT_SIZE,
            )
            format_axes(
                ax,
                title_size=SIGMA_TITLE_FONT_SIZE,
                axis_label_size=SIGMA_AXIS_LABEL_SIZE,
                tick_label_size=SIGMA_TICK_LABEL_SIZE,
            )
        fig.tight_layout()
        fig.savefig(split_output_folder(output_folder, split) / f"max_{metric}_vs_epsilon_{split}.png", dpi=300)
        plt.close(fig)


def plot_work_at_peak_reaction_vs_epsilon(
    records: list[ResultRecord],
    output_folder: Path,
    fixed_beta: float,
) -> None:
    remove_existing_plots(output_folder, "Work_at_peak_Ry_vs_epsilon_*.png")

    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            continue

        a_values = [6]
        fig, axes = plt.subplots(
            1,
            len(a_values),
            figsize=summary_figure_size(len(a_values)),
            squeeze=False,
        )
        for ax, a_value in zip(axes.ravel(), a_values):
            subset = [record for record in split_records if record.a == a_value]
            for rho in sorted({record.rho for record in subset}):
                rho_records = [
                    record
                    for record in subset
                    if math.isclose(record.rho, rho)
                ]
                for case in sorted(
                    {record.case for record in rho_records},
                    key=case_order,
                ):
                    rho_subset = sorted(
                        [
                            record
                            for record in rho_records
                            if record.case == case
                        ],
                        key=lambda record: record.epsilon,
                    )
                    if not rho_subset:
                        continue

                    ax.plot(
                        [record.epsilon for record in rho_subset],
                        [
                            normalized_metric_value(
                                "Work",
                                work_at_peak_reaction(load_graph_data(record.path)),
                            )
                            for record in rho_subset
                        ],
                        color=color_for_rho(rho),
                        linestyle=line_style_for_case(case),
                        marker="o",
                        linewidth=1.8,
                        label=rf"$\rho={rho:g}$, {case_label(case)}",
                    )
            ax.set_title(title_parameter_label(a_value, fixed_beta))
            ax.set_xlabel(length_axis_label(r"\epsilon"))
            ax.set_ylabel(r"$W(R_y=\max R_y)$ in Nmm/mm")
            ax.grid(True, alpha=0.3)
            style_legend(
                ax.legend(
                    frameon=False,
                    handlelength=3.8,
                    handletextpad=LEGEND_HANDLE_TEXT_PAD,
                    labelspacing=LEGEND_LABEL_SPACING,
                    fontsize=LEGEND_FONT_SIZE,
                )
            )
            format_axes(ax)
        fig.tight_layout()
        fig.savefig(
            split_output_folder(output_folder, split)
            / f"Work_at_peak_Ry_vs_epsilon_{split}.png",
            dpi=300,
        )
        plt.close(fig)


def plot_metric_vs_sigma_c(records: list[ResultRecord], output_folder: Path, metric: str, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, f"max_{metric}_vs_sig_c_*.png")

    records_with_sigma_c = [record for record in records if record.sigma_c is not None]
    for split in sorted({record.split for record in records_with_sigma_c}):
        split_records = [record for record in records_with_sigma_c if record.split == split]
        if not split_records:
            continue

        a_values = sorted({record.a for record in split_records})
        fig, axes = plt.subplots(1, len(a_values), figsize=(6.3 * len(a_values), 5.2), squeeze=False)
        for ax, a_value in zip(axes.ravel(), a_values):
            subset = [record for record in split_records if record.a == a_value]
            for rho in sorted({record.rho for record in subset}):
                rho_records = [record for record in subset if math.isclose(record.rho, rho)]
                for case in sorted({record.case for record in rho_records}, key=case_order):
                    sigma_subset = sorted(
                        [record for record in rho_records if record.case == case and record.sigma_c is not None],
                        key=lambda record: record.sigma_c,
                    )
                    if not sigma_subset:
                        continue

                    label = rf"$\rho={rho:g}$, {case_label(case)}"
                    ax.plot(
                        [record.sigma_c for record in sigma_subset],
                        [normalized_metric_value(metric, record.max_values[metric]) for record in sigma_subset],
                        color=color_for_rho(rho),
                        linestyle=line_style_for_case(case),
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )
            ax.set_title(title_parameter_label(a_value, fixed_beta))
            ax.set_xlabel(r"$\sigma_c$ in N/mm$^2$")
            ax.set_ylabel(quantity_label(metric))
            ax.grid(True, alpha=0.3)
            style_legend(
                ax.legend(
                    frameon=False,
                    handlelength=3.8,
                    handletextpad=LEGEND_HANDLE_TEXT_PAD,
                    labelspacing=LEGEND_LABEL_SPACING,
                    fontsize=SIGMA_LEGEND_FONT_SIZE,
                ),
                font_size=SIGMA_LEGEND_FONT_SIZE,
            )
            format_axes(
                ax,
                title_size=SIGMA_TITLE_FONT_SIZE,
                axis_label_size=SIGMA_AXIS_LABEL_SIZE,
                tick_label_size=SIGMA_TICK_LABEL_SIZE,
            )
        fig.tight_layout()
        fig.savefig(split_output_folder(output_folder, split) / f"max_{metric}_vs_sig_c_{split}.png", dpi=300)
        plt.close(fig)


def plot_split_comparison(records: list[ResultRecord], output_folder: Path, metric: str, fixed_beta: float) -> None:
    by_key = {}
    for record in records:
        key = (record.case, record.beta, record.a, record.rho, record.epsilon)
        by_key.setdefault(key, {})[record.split] = record

    pairs = [(key, value) for key, value in by_key.items() if {"spectral", "volumetric"} <= set(value)]
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for case in sorted({key[0] for key, _ in pairs}, key=case_order):
        case_pairs = [(key, value) for key, value in pairs if key[0] == case]
        for key, value in case_pairs:
            rho = key[3]
            ax.scatter(
                normalized_metric_value(metric, value["volumetric"].max_values[metric]),
                normalized_metric_value(metric, value["spectral"].max_values[metric]),
                label=case_label(case) + rf", $\rho={rho:g}$",
                color=color_for_rho(rho),
                marker=marker_for_case(case),
                s=45,
                alpha=0.85,
            )
    all_values = [
        normalized_metric_value(metric, value[split].max_values[metric])
        for _, value in pairs
        for split in ("spectral", "volumetric")
    ]
    lower, upper = min(all_values), max(all_values)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(rf"volumetric {quantity_label(metric)}")
    ax.set_ylabel(rf"spectral {quantity_label(metric)}")
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    style_legend(ax.legend(unique_labels.values(), unique_labels.keys(), frameon=False, handlelength=3.0, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
    format_axes(ax)
    fig.tight_layout()
    fig.savefig(output_folder / f"split_comparison_max_{metric}.png", dpi=300)
    plt.close(fig)


def plot_volume(records: list[ResultRecord], output_folder: Path, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, "volumes_*.png")

    records_with_volume = [record for record in records if record.volume is not None]
    if not records_with_volume:
        return
    for split in sorted({record.split for record in records_with_volume}):
        split_records = [record for record in records_with_volume if record.split == split]
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        sorted_records = sorted(split_records, key=lambda record: (case_order(record.case), record.epsilon, record.a, record.rho))
        for case in sorted({record.case for record in sorted_records}, key=case_order):
            case_records = [record for record in sorted_records if record.case == case]
            ax.scatter(
                [sorted_records.index(record) for record in case_records],
                [record.volume for record in case_records],
                label=case_label(case),
                color=[color_for_rho(record.rho) for record in case_records],
                marker=marker_for_case(case),
                s=42,
            )
        ax.set_ylabel(r"$V$")
        ax.set_xlabel(r"parameter-combination index")
        ax.grid(True, alpha=0.3)
        style_legend(ax.legend(frameon=False, handlelength=3.0, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
        format_axes(ax)
        fig.tight_layout()
        fig.savefig(split_output_folder(output_folder, split) / f"volumes_{split}.png", dpi=300)
        plt.close(fig)


def plot_volume_vs_rho_a(records: list[ResultRecord], output_folder: Path, fixed_beta: float) -> None:
    remove_existing_plots(output_folder, "volumes_vs_rho_a_*.png")

    records_with_volume = [record for record in records if record.volume is not None]
    if not records_with_volume:
        return
    for split in sorted({record.split for record in records_with_volume}):
        split_records = [record for record in records_with_volume if record.split == split]
        epsilon_values = sorted({record.epsilon for record in split_records})
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            for epsilon in epsilon_values:
                epsilon_records = sorted(
                    [
                        record
                        for record in case_records
                        if math.isclose(record.epsilon, epsilon)
                    ],
                    key=lambda record: record.rho * record.a,
                )
                if not epsilon_records:
                    continue

                x_values = [record.rho * record.a for record in epsilon_records]
                y_values = [record.volume for record in epsilon_records]
                ax.plot(
                    x_values,
                    y_values,
                    color="#7f8790",
                    linestyle=line_style_for_case(case),
                    linewidth=1.5,
                    alpha=0.65,
                    label=case_label(case) + ", " + length_label(r"\epsilon", epsilon),
                )
                for record, x_value, y_value in zip(epsilon_records, x_values, y_values):
                    ax.scatter(
                        x_value,
                        y_value,
                        color=color_for_rho(record.rho),
                        marker=marker_for_epsilon(record.epsilon, epsilon_values),
                        edgecolor="black",
                        linewidth=0.35,
                        s=48,
                        zorder=3,
                    )
        ax.set_ylabel(r"$V$")
        ax.set_xlabel(r"$\rho a\,\mathrm{[mm]}$")
        ax.grid(True, alpha=0.3)
        style_legend(ax.legend(frameon=False, handlelength=3.0, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, ncol=2, fontsize=LEGEND_FONT_SIZE))
        format_axes(ax)
        fig.tight_layout()
        fig.savefig(split_output_folder(output_folder, split) / f"volumes_vs_rho_a_{split}.png", dpi=300)
        plt.close(fig)


def plot_porosity_volume_vs_density_a(
    records: list[ResultRecord],
    output_folder: Path,
    fixed_beta: float,
) -> None:
    remove_existing_plots(output_folder, "porosity_volume_vs_density_a_*.png")

    records_with_porosity = [
        record
        for record in records
        if record.volume is not None and record.porosity_average is not None
    ]
    if not records_with_porosity:
        return
    for split in sorted({record.split for record in records_with_porosity}):
        split_records = [record for record in records_with_porosity if record.split == split]
        epsilon_values = sorted({record.epsilon for record in split_records})
        fig, ax = plt.subplots(figsize=(9.2, 5.8))
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            for epsilon in epsilon_values:
                epsilon_records = sorted(
                    [
                        record
                        for record in case_records
                        if math.isclose(record.epsilon, epsilon)
                    ],
                    key=lambda record: record.rho * record.a,
                )
                if not epsilon_records:
                    continue

                x_values = [record.rho * record.a for record in epsilon_records]
                y_values = [
                    record.porosity_average * record.volume
                    for record in epsilon_records
                    if record.porosity_average is not None and record.volume is not None
                ]
                if len(y_values) != len(x_values):
                    continue

                ax.plot(
                    x_values,
                    y_values,
                    color="#7f8790",
                    linestyle=line_style_for_case(case),
                    linewidth=1.5,
                    alpha=0.65,
                    label=case_label(case) + ", " + length_label(r"\epsilon", epsilon),
                )
                for record, x_value, y_value in zip(epsilon_records, x_values, y_values):
                    ax.scatter(
                        x_value,
                        y_value,
                        color=color_for_rho(record.rho),
                        marker=marker_for_epsilon(record.epsilon, epsilon_values),
                        edgecolor="black",
                        linewidth=0.35,
                        s=48,
                        zorder=3,
                    )
        ax.set_ylabel(r"$\overline{\phi} V$")
        ax.set_xlabel(r"$\rho a\,\mathrm{[mm]}$")
        ax.grid(True, alpha=0.3)
        style_legend(ax.legend(frameon=False, handlelength=3.0, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, ncol=2, fontsize=LEGEND_FONT_SIZE))
        format_axes(ax)
        fig.tight_layout()
        fig.savefig(
            split_output_folder(output_folder, split)
            / f"porosity_volume_vs_density_a_{split}.png",
            dpi=300,
        )
        plt.close(fig)


def main() -> None:
    global GC_REFERENCE, MU_REFERENCE, OMIT_A_IN_TITLES, REFERENCE_LENGTH, SHOW_PARAMETER_TITLES

    args = parse_args()
    if args.reference_length <= 0.0:
        raise SystemExit("--reference-length must be positive.")
    if args.reference_gc <= 0.0:
        raise SystemExit("--reference-gc must be positive.")
    if args.reference_mu <= 0.0:
        raise SystemExit("--reference-mu must be positive.")
    REFERENCE_LENGTH = args.reference_length
    GC_REFERENCE = args.reference_gc
    MU_REFERENCE = args.reference_mu
    OMIT_A_IN_TITLES = args.omit_a_in_titles
    SHOW_PARAMETER_TITLES = args.show_parameter_titles

    script_path = Path(__file__).resolve().parent
    result_roots = args.result_roots or [script_path / "results"]
    output_folder = args.output_folder or script_path / "plots" / "260504_evaluation_plots"
    output_folder.mkdir(parents=True, exist_ok=True)

    records = collect_records(result_roots)
    records = filter_records(records, args.splits, args.a_values)
    if not records:
        raise SystemExit("No matching result_graphs_*.txt files found.")

    fixed_beta = select_fixed_beta(records, args.fixed_beta)
    fixed_beta_records = filter_by_beta(records, fixed_beta)

    write_summary_csv(records, output_folder)
    plot_curves(fixed_beta_records, output_folder, args.x_limit, fixed_beta)
    for metric in QUANTITIES:
        plot_metric_vs_beta(records, output_folder, metric)
        plot_metric_vs_rho(fixed_beta_records, output_folder, metric, fixed_beta)
        plot_metric_vs_epsilon(fixed_beta_records, output_folder, metric, fixed_beta)
        plot_metric_vs_sigma_c(fixed_beta_records, output_folder, metric, fixed_beta)
        plot_split_comparison(fixed_beta_records, output_folder, metric, fixed_beta)
    plot_work_at_peak_reaction_vs_epsilon(fixed_beta_records, output_folder, fixed_beta)
    plot_volume(fixed_beta_records, output_folder, fixed_beta)
    plot_volume_vs_rho_a(fixed_beta_records, output_folder, fixed_beta)
    plot_porosity_volume_vs_density_a(fixed_beta_records, output_folder, fixed_beta)

    print(f"Wrote plots for {len(records)} result files to {output_folder}")
    print(f"Used beta_phi={fixed_beta:g} for all non-vs-beta_phi plots.")
    print(
        f"Used reference scales length={REFERENCE_LENGTH:g} mm, "
        f"G_c^0={GC_REFERENCE:g}, mu^0={MU_REFERENCE:g}."
    )


if __name__ == "__main__":
    main()
