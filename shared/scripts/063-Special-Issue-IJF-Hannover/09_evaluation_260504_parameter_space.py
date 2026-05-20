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
        "max_label": r"$\max \Pi_\mathrm{frac}$ in Nmm/mm",
    },
    "Elastic": {
        "column": 6,
        "curve_label": r"$\Pi_\mathrm{el}$ in Nmm/mm",
        "max_label": r"$\max \Pi_\mathrm{el}$ in Nmm/mm",
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
AXIS_LABEL_SIZE = 22
TICK_LABEL_SIZE = 17
LEGEND_FONT_SIZE = 17
TITLE_FONT_SIZE = 20
SUPTITLE_FONT_SIZE = 22
LEGEND_HANDLE_TEXT_PAD = 1.25
LEGEND_LABEL_SPACING = 0.55
AXIS_LABEL_PAD = 9
POISSON_RATIO = 0.3
E_REFERENCE = 210000.0
GC_REFERENCE = 1.0
MU_REFERENCE = E_REFERENCE / (2.0 * (1.0 + POISSON_RATIO))
REFERENCE_LENGTH = 1.0
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
        help="Optional maximum displacement in mm shown in curve plots. Defaults to an automatic per-plot limit.",
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
            max_values = {
                name: float(np.max(np.abs(data[:, spec["column"]])))
                for name, spec in QUANTITIES.items()
            }
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


def style_legend(legend) -> None:
    if legend is None:
        return
    legend.set_title(legend.get_title().get_text(), prop={"size": LEGEND_FONT_SIZE})
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


def format_axes(ax: plt.Axes, title_size: int = TITLE_FONT_SIZE) -> None:
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.xaxis.labelpad = AXIS_LABEL_PAD
    ax.yaxis.labelpad = AXIS_LABEL_PAD
    ax.title.set_size(title_size)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


def format_suptitle(fig: plt.Figure) -> None:
    if fig._suptitle is not None:
        fig._suptitle.set_size(SUPTITLE_FONT_SIZE)


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
                ax.set_title(
                    length_label("a", a_value) + ", "
                    + length_label(r"\epsilon", epsilon)
                    + fixed_beta_label(fixed_beta)
                )
                ax.set_xlabel(r"$u_y$ in mm")
                ax.set_ylabel(quantity_label(quantity, "curve_label"))
                ax.set_xlim(0.0, x_limit or nice_axis_upper_limit(max_displacement))
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
                ax.set_title(length_label("a", a_value) + fixed_beta_label(fixed_beta))
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
        fig, axes = plt.subplots(1, len(a_values), figsize=(6.3 * len(a_values), 5.2), squeeze=False)
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
            ax.set_title(length_label("a", a_value) + fixed_beta_label(fixed_beta))
            ax.set_xlabel(length_axis_label(r"\epsilon"))
            ax.set_ylabel(quantity_label(metric))
            ax.grid(True, alpha=0.3)
            style_legend(ax.legend(frameon=False, handlelength=3.8, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
            format_axes(ax)
        fig.tight_layout()
        fig.savefig(split_output_folder(output_folder, split) / f"max_{metric}_vs_epsilon_{split}.png", dpi=300)
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
                        [record.sigma_c / stress_scale() for record in sigma_subset],
                        [normalized_metric_value(metric, record.max_values[metric]) for record in sigma_subset],
                        color=color_for_rho(rho),
                        linestyle=line_style_for_case(case),
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )
            ax.set_title(length_label("a", a_value) + fixed_beta_label(fixed_beta))
            ax.set_xlabel(r"$\sigma_c/\sqrt{2\mu^0 G_c^0/\mathrm{mm}}$")
            ax.set_ylabel(quantity_label(metric))
            ax.grid(True, alpha=0.3)
            style_legend(ax.legend(frameon=False, handlelength=3.8, handletextpad=LEGEND_HANDLE_TEXT_PAD, labelspacing=LEGEND_LABEL_SPACING, fontsize=LEGEND_FONT_SIZE))
            format_axes(ax)
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
    global GC_REFERENCE, MU_REFERENCE, REFERENCE_LENGTH

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
