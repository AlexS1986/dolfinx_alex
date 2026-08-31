#!/usr/bin/env python3
"""
Copy the Neper-generated meshes + orientation tables into ./inputs/ so the
dolfinx scripts (homogenize_rve.py, uniaxial_tension.py) can read them.

Run on the HOST (where both the Neper folder and this folder are visible) -
NOT inside a container. It copies each .xdmf together with its .h5 (the .xdmf
references the .h5 by relative name, so both must sit in the same folder).

Examples:
  # RVE cubes for homogenization (both steels, n=300):
  python3 prepare_inputs.py --rve 316L 17-4PH --n 300

  # directional tensile bars (step 7) for the uniaxial test:
  python3 prepare_inputs.py --specimens 316L 17-4PH

  # custom Neper folder:
  python3 prepare_inputs.py --neper-dir /path/to/neper_pipeline --rve 316L --n 300
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NEPER = os.path.normpath(os.path.join(
    HERE, "..", "..", "..", "..", "Meshing", "Neper",
    "data", "04_anisotropy_waam", "neper_pipeline"))
ORIENTS = ["V", "H", "45deg"]


def copy(src, dst):
    if not os.path.isfile(src):
        print(f"  MISSING: {src}")
        return False
    shutil.copy2(src, dst)
    print(f"  copied {os.path.basename(src)}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neper-dir", default=DEFAULT_NEPER)
    ap.add_argument("--n", type=int, default=300, help="grain count of the RVE meshes")
    ap.add_argument("--rve", nargs="*", default=[], help="materials: RVE cube + grain_ori")
    ap.add_argument("--specimens", nargs="*", default=[],
                    help="materials: V/H/45deg bars + grain_ori")
    args = ap.parse_args()

    nd = args.neper_dir
    out = os.path.join(HERE, "inputs")
    os.makedirs(out, exist_ok=True)
    print(f"Neper dir : {nd}")
    print(f"inputs -> : {out}")
    if not os.path.isdir(nd):
        raise SystemExit(f"Neper folder not found: {nd} (use --neper-dir)")

    for mat in args.rve:
        base = f"waam_{mat}_n{args.n}"
        print(f"[RVE {mat}]")
        for ext in (".xdmf", ".h5"):
            copy(os.path.join(nd, base + ext), os.path.join(out, base + ext))
        copy(os.path.join(nd, f"grain_ori_{mat}.txt"),
             os.path.join(out, f"grain_ori_{mat}.txt"))

    for mat in args.specimens:
        print(f"[specimens {mat}]")
        for ori in ORIENTS:
            base = f"spec_{mat}_{ori}"
            for ext in (".xdmf", ".h5"):
                copy(os.path.join(nd, base + ext), os.path.join(out, base + ext))
            copy(os.path.join(nd, f"grain_ori_{mat}_{ori}.txt"),
                 os.path.join(out, f"grain_ori_{mat}_{ori}.txt"))

    print("\ndone. Next (in the dolfinx container): run homogenize_rve.py / uniaxial_tension.py")


if __name__ == "__main__":
    main()
