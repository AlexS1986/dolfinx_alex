#!/usr/bin/env python3
"""Plot the volume-constraint integral from XDMF Young's modulus fields."""

from __future__ import annotations

import argparse
import csv
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
    r"^results_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_"
    r"(?P<rho>0_\d+)_(?P<structure>min|max|var)_"
    r"(?P<case>min|max|vary)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_]+))?\.xdmf$"
)
SIMULATION_FOLDER_RE = re.compile(
    r"^simulation_.*_SPLIT(?P<split>spectral|volumetric)"
    r"_EPS(?P<epsilon>[0-9_]+)$"
)

E0 = 126000.0
E1 = 210000.0
Q = 2.0
PHI_MIN = 0.5
DESIGN_AREA = 6.0


@dataclass(frozen=True)
class ConstraintRecord:
    path: Path
    case: str
    rho: float
    area: float
    rho_omega: float
    target: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_root",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "plots"
            / "reduced_spectral_a6"
            / "rho_omega_constraint.pdf"
        ),
    )
    parser.add_argument("--fixed-beta", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=0.015)
    parser.add_argument("--a-value", type=int, default=6)
    parser.add_argument("--split", default="spectral")
    return parser.parse_args()


def token_to_float(token: str) -> float:
    return float(token.replace("_", "."))


def epsilon_from_path(path: Path, match: re.Match[str]) -> float:
    if match.group("epsilon"):
        return token_to_float(match.group("epsilon"))
    for parent in path.parents:
        folder_match = SIMULATION_FOLDER_RE.match(parent.name)
        if folder_match:
            return token_to_float(folder_match.group("epsilon"))
    return math.inf


def matching_xdmf_files(args: argparse.Namespace) -> list[Path]:
    selected = []
    for path in args.result_root.rglob("results_*.xdmf"):
        match = RESULT_RE.match(path.name)
        if not match:
            continue
        beta = token_to_float(match.group("beta"))
        epsilon = epsilon_from_path(path, match)
        if not math.isclose(beta, args.fixed_beta):
            continue
        if not math.isclose(epsilon, args.epsilon):
            continue
        if int(match.group("a")) != args.a_value:
            continue
        if match.group("split") != args.split:
            continue
        selected.append(path)
    return sorted(selected, key=sort_key)


def sort_key(path: Path) -> tuple[int, float]:
    match = RESULT_RE.match(path.name)
    mode_order = {"min": 0, "max": 1, "vary": 2}
    case = match.group("case") if match else ""
    rho = token_to_float(match.group("rho")) if match else math.inf
    return mode_order.get(case, 99), rho


def triangle_areas(points: np.ndarray, cells: np.ndarray) -> np.ndarray:
    p0 = points[cells[:, 0], :2]
    p1 = points[cells[:, 1], :2]
    p2 = points[cells[:, 2], :2]
    return 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )


def load_first_E_field(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h5_path = path.with_suffix(".h5")
    with h5py.File(h5_path, "r") as h5:
        points = np.asarray(h5["/Mesh/mesh/geometry"])
        cells = np.asarray(h5["/Mesh/mesh/topology"], dtype=np.int64)
        field_group = h5["/Function/E"]
        time_key = min(field_group.keys(), key=lambda key: float(key.replace("_", ".")))
        E_values = np.asarray(field_group[time_key]).reshape(-1)
    return points, cells, E_values


def phi_from_E(E_values: np.ndarray) -> np.ndarray:
    normalized = np.clip((E_values - E0) / (E1 - E0), 0.0, 1.0)
    return normalized ** (1.0 / Q)


def integrate_constraint(path: Path) -> ConstraintRecord:
    match = RESULT_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected result name: {path.name}")
    rho = token_to_float(match.group("rho"))
    points, cells, E_values = load_first_E_field(path)
    areas = triangle_areas(points, cells)
    phi = phi_from_E(E_values)
    phi_p = (1.0 - PHI_MIN) * phi + PHI_MIN
    cell_phi_p = np.mean(phi_p[cells], axis=1)
    rho_omega = float(np.sum(cell_phi_p * areas))
    return ConstraintRecord(
        path=path,
        case=match.group("case"),
        rho=rho,
        area=float(np.sum(areas)),
        rho_omega=rho_omega,
        target=rho * DESIGN_AREA,
    )


def display_case(case: str) -> str:
    return {"min": "min", "max": "max", "vary": "vary"}[case]


def plot_records(records: list[ConstraintRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        rf"$\mathrm{{{display_case(record.case)}}}$" + "\n"
        + rf"$\rho={record.rho:g}$"
        for record in records
    ]
    values = [record.rho_omega for record in records]
    targets = [record.target for record in records]
    colors = ["#c83f49" if math.isclose(record.rho, 0.3) else "#2f6fb7" for record in records]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(records))
    ax.bar(x, values, color=colors, width=0.68, label=r"integral")
    ax.scatter(
        x,
        targets,
        marker="_",
        s=520,
        color="black",
        linewidth=2.4,
        label=r"$\rho\Omega$, $\Omega=6\,\mathrm{mm}^2$",
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel(r"$\rho\Omega$ in mm$^2$", fontsize=18)
    ax.tick_params(axis="y", labelsize=15)
    ax.grid(True, axis="y", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def write_csv(records: list[ConstraintRecord], output: Path) -> None:
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "rho", "area", "rho_omega", "target", "path"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({
                "case": record.case,
                "rho": record.rho,
                "area": record.area,
                "rho_omega": record.rho_omega,
                "target": record.target,
                "path": record.path,
            })


def main() -> None:
    args = parse_args()
    records = [integrate_constraint(path) for path in matching_xdmf_files(args)]
    if not records:
        raise SystemExit("No matching XDMF files found.")
    plot_records(records, args.output)
    write_csv(records, args.output)
    for record in records:
        print(
            f"{record.case:>4s} rho={record.rho:g}: "
            f"rhoOmega={record.rho_omega:.6f}, target={record.target:.6f}"
        )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.png')}")
    print(f"Wrote {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
