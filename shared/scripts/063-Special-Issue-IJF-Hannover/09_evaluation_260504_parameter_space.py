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

import matplotlib.pyplot as plt
import numpy as np


RESULT_RE = re.compile(
    r"^result_graphs_beta_(?P<beta>0_\d+)_a_(?P<a>\d+)_rho_(?P<rho>0_\d+)"
    r"_(?P<structure>min|max|var)_(?P<case>min|max|vary)_(?P<split>spectral|volumetric)"
    r"(?:_eps(?P<epsilon>[0-9_m]+))?\.txt$"
)

QUANTITIES = {
    "Ry": {"column": 2, "label": r"max $R_y$ / (N/mm)"},
    "Work": {"column": 4, "label": r"max work / mm"},
    "Fracture": {"column": 5, "label": r"max fracture energy / mm"},
    "Elastic": {"column": 6, "label": r"max elastic energy / mm"},
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 260504 DCB parameter-space plots from result_graphs files."
    )
    parser.add_argument(
        "resource_roots",
        nargs="*",
        type=Path,
        default=[Path("resources")],
        help="Resource roots to scan recursively. Defaults to ./resources.",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=Path("resources/260504_evaluation_plots"),
        help="Folder for generated plots and summary CSV.",
    )
    parser.add_argument(
        "--x-limit",
        type=float,
        default=0.05,
        help="Maximum displacement shown in curve plots.",
    )
    return parser.parse_args()


def parse_float(token: str) -> float:
    return float(token.replace("_", ".").replace("m", "-"))


def parse_result_path(path: Path) -> dict[str, object] | None:
    match = RESULT_RE.match(path.name)
    if not match:
        return None
    return {
        "beta": parse_float(match.group("beta")),
        "a": int(match.group("a")),
        "rho": parse_float(match.group("rho")),
        "case": match.group("case"),
        "split": match.group("split"),
        "epsilon": parse_float(match.group("epsilon")) if match.group("epsilon") else 0.015,
    }


def load_graph_data(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as infile:
        lines = [line for line in infile if line.strip() and not line.startswith("#")]
    data = np.loadtxt(lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def find_volume(path: Path, case: str, split: str) -> float | None:
    candidates = sorted(path.parent.glob(f"vol_*_{case}_{split}*.json"))
    if not candidates:
        return None
    with candidates[0].open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    return data.get("vol")


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
                    volume=find_volume(path, parsed["case"], parsed["split"]),
                )
            )
    return sorted(records, key=record_sort_key)


def record_sort_key(record: ResultRecord) -> tuple[str, str, int, float, float, float]:
    return (record.split, record.case, record.a, record.beta, record.rho, record.epsilon)


def label_for_record(record: ResultRecord) -> str:
    label = rf"$\beta={record.beta:g}$, $a={record.a}$, $\rho={record.rho:g}$"
    if not math.isclose(record.epsilon, 0.015):
        label += rf", $\epsilon={record.epsilon:g}$"
    return label


def case_order(case: str) -> int:
    return {"min": 0, "max": 1, "vary": 2}.get(case, 99)


def write_summary_csv(records: list[ResultRecord], output_folder: Path) -> None:
    path = output_folder / "summary_260504.csv"
    fieldnames = ["split", "case", "beta", "a", "rho", "epsilon", "volume"]
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
            }
            row.update({f"max_{name}": value for name, value in record.max_values.items()})
            row.update({f"final_{name}": value for name, value in record.final_values.items()})
            writer.writerow(row)


def plot_curves(records: list[ResultRecord], output_folder: Path, x_limit: float) -> None:
    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            for quantity, spec in QUANTITIES.items():
                fig, ax = plt.subplots(figsize=(9, 5.5))
                for record in case_records:
                    data = load_graph_data(record.path)
                    displacement = np.abs(data[:, 1])
                    values = np.abs(data[:, spec["column"]])
                    ax.plot(displacement, values, label=label_for_record(record), linewidth=1.6)
                ax.set_xlabel(r"$u_y$ / mm")
                ax.set_ylabel(spec["label"].replace("max ", ""))
                ax.set_xlim(0.0, x_limit)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, ncol=2, frameon=False)
                ax.set_title(f"{quantity} vs displacement, {split}, {case}")
                fig.tight_layout()
                fig.savefig(output_folder / f"{quantity}_vs_uy_{split}_{case}.png", dpi=300)
                plt.close(fig)


def plot_metric_vs_beta(records: list[ResultRecord], output_folder: Path, metric: str) -> None:
    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            if not case_records:
                continue

            a_values = sorted({record.a for record in case_records})
            fig, axes = plt.subplots(1, len(a_values), figsize=(5.2 * len(a_values), 4.4), squeeze=False)
            for ax, a_value in zip(axes.ravel(), a_values):
                subset = [record for record in case_records if record.a == a_value]
                for rho, epsilon in sorted({(record.rho, record.epsilon) for record in subset}):
                    rho_subset = sorted(
                        [
                            record
                            for record in subset
                            if record.rho == rho and math.isclose(record.epsilon, epsilon)
                        ],
                        key=lambda record: record.beta,
                    )
                    label = rf"$\rho={rho:g}$"
                    if not math.isclose(epsilon, 0.015):
                        label += rf", $\epsilon={epsilon:g}$"
                    ax.plot(
                        [record.beta for record in rho_subset],
                        [record.max_values[metric] for record in rho_subset],
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )
                ax.set_xscale("log")
                ax.set_xlabel(r"$\beta$")
                ax.set_ylabel(QUANTITIES[metric]["label"])
                ax.set_title(rf"$a={a_value}$")
                ax.grid(True, alpha=0.3)
                ax.legend(frameon=False)
            fig.suptitle(f"{metric}: beta dependence ({split}, {case})")
            fig.tight_layout()
            fig.savefig(output_folder / f"max_{metric}_vs_beta_{split}_{case}.png", dpi=300)
            plt.close(fig)


def plot_metric_vs_rho(records: list[ResultRecord], output_folder: Path, metric: str) -> None:
    for split in sorted({record.split for record in records}):
        split_records = [record for record in records if record.split == split]
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = [record for record in split_records if record.case == case]
            if not case_records:
                continue

            a_values = sorted({record.a for record in case_records})
            fig, axes = plt.subplots(1, len(a_values), figsize=(5.2 * len(a_values), 4.4), squeeze=False)
            for ax, a_value in zip(axes.ravel(), a_values):
                subset = [record for record in case_records if record.a == a_value]
                for beta, epsilon in sorted({(record.beta, record.epsilon) for record in subset}):
                    beta_subset = sorted(
                        [
                            record
                            for record in subset
                            if math.isclose(record.beta, beta) and math.isclose(record.epsilon, epsilon)
                        ],
                        key=lambda record: record.rho,
                    )
                    label = rf"$\beta={beta:g}$"
                    if not math.isclose(epsilon, 0.015):
                        label += rf", $\epsilon={epsilon:g}$"
                    ax.plot(
                        [record.rho for record in beta_subset],
                        [record.max_values[metric] for record in beta_subset],
                        marker="o",
                        linewidth=1.8,
                        label=label,
                    )
                ax.set_xlabel(r"$\rho$")
                ax.set_ylabel(QUANTITIES[metric]["label"])
                ax.set_title(rf"$a={a_value}$")
                ax.grid(True, alpha=0.3)
                ax.legend(frameon=False)
            fig.suptitle(f"{metric}: density dependence ({split}, {case})")
            fig.tight_layout()
            fig.savefig(output_folder / f"max_{metric}_vs_rho_{split}_{case}.png", dpi=300)
            plt.close(fig)


def plot_split_comparison(records: list[ResultRecord], output_folder: Path, metric: str) -> None:
    by_key = {}
    for record in records:
        key = (record.case, record.beta, record.a, record.rho, record.epsilon)
        by_key.setdefault(key, {})[record.split] = record

    pairs = [(key, value) for key, value in by_key.items() if {"spectral", "volumetric"} <= set(value)]
    if not pairs:
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    for case in sorted({key[0] for key, _ in pairs}, key=case_order):
        case_pairs = [(key, value) for key, value in pairs if key[0] == case]
        ax.scatter(
            [value["volumetric"].max_values[metric] for _, value in case_pairs],
            [value["spectral"].max_values[metric] for _, value in case_pairs],
            label=case,
            s=45,
            alpha=0.85,
        )
    all_values = [
        value[split].max_values[metric]
        for _, value in pairs
        for split in ("spectral", "volumetric")
    ]
    lower, upper = min(all_values), max(all_values)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel(f"volumetric {QUANTITIES[metric]['label']}")
    ax.set_ylabel(f"spectral {QUANTITIES[metric]['label']}")
    ax.set_title(f"Spectral vs volumetric: {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_folder / f"split_comparison_max_{metric}.png", dpi=300)
    plt.close(fig)


def plot_volume(records: list[ResultRecord], output_folder: Path) -> None:
    records_with_volume = [record for record in records if record.volume is not None]
    if not records_with_volume:
        return
    for split in sorted({record.split for record in records_with_volume}):
        split_records = [record for record in records_with_volume if record.split == split]
        fig, ax = plt.subplots(figsize=(8, 5))
        for case in sorted({record.case for record in split_records}, key=case_order):
            case_records = sorted(
                [record for record in split_records if record.case == case],
                key=lambda record: (record.beta, record.a, record.rho),
            )
            ax.scatter(
                np.arange(len(case_records)),
                [record.volume for record in case_records],
                label=case,
                s=42,
            )
        ax.set_ylabel("Volume")
        ax.set_xlabel("parameter combination index")
        ax.set_title(f"Volume overview ({split})")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_folder / f"volumes_{split}.png", dpi=300)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    records = collect_records(args.resource_roots)
    if not records:
        raise SystemExit("No matching result_graphs_*.txt files found.")

    write_summary_csv(records, args.output_folder)
    plot_curves(records, args.output_folder, args.x_limit)
    for metric in QUANTITIES:
        plot_metric_vs_beta(records, args.output_folder, metric)
        plot_metric_vs_rho(records, args.output_folder, metric)
        plot_split_comparison(records, args.output_folder, metric)
    plot_volume(records, args.output_folder)

    print(f"Wrote plots for {len(records)} result files to {args.output_folder}")


if __name__ == "__main__":
    main()
