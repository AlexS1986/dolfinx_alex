#!/usr/bin/env python3
"""Run the CT-scan phase-field simulation with linear finite elements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import alex.postprocessing as pp
from mpi4py import MPI

import pfmfrac_function as simulation


MATERIAL_PRESETS = {
    # Same E [MPa] and nu values as pygalmesh/009 linearelastic.py.
    "am": (73000.0, 0.36),
    "std": (70000.0, 0.35),
    "conv": (82000.0, 0.35),
    "ad": (82000.0, 0.35),
}


def lame_from_engineering_constants(youngs_modulus: float, poisson_ratio: float):
    if youngs_modulus <= 0.0:
        raise ValueError("Young's modulus must be positive.")
    if not -1.0 < poisson_ratio < 0.5:
        raise ValueError("Poisson's ratio must lie strictly between -1 and 0.5.")
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / (
        (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
    )
    return lam, mu


def engineering_constants_from_lame(lam: float, mu: float):
    if mu <= 0.0 or 3.0 * lam + 2.0 * mu <= 0.0:
        raise ValueError(
            "Lamé parameters must satisfy mu > 0 and 3*lambda + 2*mu > 0."
        )
    poisson_ratio = lam / (2.0 * (lam + mu))
    youngs_modulus = mu * (3.0 * lam + 2.0 * mu) / (lam + mu)
    return youngs_modulus, poisson_ratio


def resolve_material_parameters(args: argparse.Namespace) -> dict[str, float | str]:
    has_lame = args.lam is not None or args.mu is not None
    has_engineering = (
        args.youngs_modulus is not None or args.poisson_ratio is not None
    )

    if args.material != "custom":
        if has_lame or has_engineering:
            raise ValueError(
                "Do not combine a material preset with --lam/--mu or "
                "--youngs-modulus/--poisson-ratio."
            )
        youngs_modulus, poisson_ratio = MATERIAL_PRESETS[args.material]
        lam, mu = lame_from_engineering_constants(youngs_modulus, poisson_ratio)
        source = "pygalmesh_009_preset"
        unit = "MPa"
    elif has_lame:
        if args.lam is None or args.mu is None:
            raise ValueError("--lam and --mu must be provided together.")
        if has_engineering:
            raise ValueError(
                "Choose either --lam/--mu or --youngs-modulus/--poisson-ratio."
            )
        lam, mu = args.lam, args.mu
        youngs_modulus, poisson_ratio = engineering_constants_from_lame(lam, mu)
        source = "direct_lame"
        unit = "user_consistent_unit"
    elif has_engineering:
        if args.youngs_modulus is None or args.poisson_ratio is None:
            raise ValueError(
                "--youngs-modulus and --poisson-ratio must be provided together."
            )
        youngs_modulus, poisson_ratio = (
            args.youngs_modulus,
            args.poisson_ratio,
        )
        lam, mu = lame_from_engineering_constants(youngs_modulus, poisson_ratio)
        source = "youngs_modulus_and_poisson_ratio"
        unit = "user_consistent_unit"
    else:
        lam = 1.0
        mu = 1.0
        youngs_modulus, poisson_ratio = engineering_constants_from_lame(lam, mu)
        source = "legacy_default_lame"
        unit = "unspecified_consistent_unit"

    return {
        "material": args.material,
        "material_parameter_source": source,
        "elastic_parameter_unit": unit,
        "youngs_modulus": youngs_modulus,
        "poisson_ratio": poisson_ratio,
        "lam": lam,
        "mu": mu,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 3D CT-scan phase-field fracture simulation. The mesh "
            "must be available as <mesh-base>.xdmf with its referenced HDF5 file."
        )
    )
    parser.add_argument("--mesh-base", required=True)
    parser.add_argument(
        "--material",
        choices=("custom", *MATERIAL_PRESETS),
        default="custom",
        help="material preset from pygalmesh/009, or custom (default)",
    )
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--youngs-modulus", type=float, default=None)
    parser.add_argument("--poisson-ratio", type=float, default=None)
    parser.add_argument("--gc", type=float, default=1.0)
    parser.add_argument("--epsilon-factor", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    material = resolve_material_parameters(args)
    metadata = {
        "mesh_base": args.mesh_base,
        **material,
        "gc": args.gc,
        "epsilon_factor": args.epsilon_factor,
        "element_order": 1,
        "mpi_processes": MPI.COMM_WORLD.size,
    }
    parameters_to_write = {
        "mesh_file": args.mesh_base,
        "material": material["material"],
        "material_parameter_source": material["material_parameter_source"],
        "elastic_parameter_unit": material["elastic_parameter_unit"],
        "E_mod_simulation": material["youngs_modulus"],
        "nu_simulation": material["poisson_ratio"],
        "lam_simulation": material["lam"],
        "mue_simulation": material["mu"],
        "Gc_simulation": args.gc,
        "eps_factor_simulation": args.epsilon_factor,
        "element_order": 1,
        "mpi_processes": MPI.COMM_WORLD.size,
    }
    pp.write_to_file(
        filename="parameters.txt",
        parameters=parameters_to_write,
        comm=MPI.COMM_WORLD,
    )
    if MPI.COMM_WORLD.rank == 0:
        Path("run_parameters.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            "Resolved elastic parameters: "
            f"material={material['material']}, E={material['youngs_modulus']}, "
            f"nu={material['poisson_ratio']}, lambda={material['lam']}, "
            f"mu={material['mu']}"
        )
    simulation.run_simulation(
        mesh_file=args.mesh_base,
        lam_param=material["lam"],
        mue_param=material["mu"],
        Gc_param=args.gc,
        eps_factor_param=args.epsilon_factor,
        element_order=1,
        comm=MPI.COMM_WORLD,
    )


if __name__ == "__main__":
    main()
