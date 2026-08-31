#!/usr/bin/env python3
"""
Extract the full orthotropic engineering constants from the homogenized
stiffness tensors Chom_<mat>.json (KUBC), via the compliance S = C^-1.

For each material writes the 9 orthotropic constants (E_x,E_y,E_z; the six
Poisson ratios nu_ij; the three shear moduli G_yz,G_xz,G_xy) in the RVE frame
x = weld, y = wall-normal, z = build (Voigt order xx,yy,zz,yz,xz,xy, engineering
shear). These are the constants to quote in a publication, taken from the full
inverse tensor rather than from single stiffness components.

Output:
  report/engineering_constants.md   (markdown table, for the report)
  engineering_constants.csv         (machine-readable)

Note on reciprocity: because S is symmetric, nu_ij/E_i == nu_ji/E_j holds
exactly by construction; the residual normal-shear coupling max|C[0:3,3:6]|
quantifies how far the finite-N RVE is from perfect orthotropy.

Usage:  python3 engineering_constants.py [--materials 316L 17-4PH]
"""
import argparse
import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = ("x=weld", "y=wall-normal", "z=build")


def constants_from_C(C):
    """Orthotropic engineering constants from a 6x6 stiffness C [GPa]."""
    S = np.linalg.inv(np.asarray(C, float))
    E = [1.0 / S[i, i] for i in range(3)]
    G = [1.0 / S[3, 3], 1.0 / S[4, 4], 1.0 / S[5, 5]]   # yz, xz, xy
    # nu_ij = lateral contraction in j under uniaxial stress in i = -S[j,i]/S[i,i]
    nu = {(i, j): -S[j, i] / S[i, i] for i in range(3) for j in range(3) if i != j}
    coupling = float(np.max(np.abs(np.asarray(C)[:3, 3:])))  # normal-shear coupling
    return dict(E=E, G=G, nu=nu, coupling=coupling)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--materials", nargs="*", default=["316L", "17-4PH"])
    ap.add_argument("--dir", default=HERE)
    args = ap.parse_args()

    rows = []
    for mat in args.materials:
        path = os.path.join(args.dir, f"Chom_{mat}.json")
        if not os.path.exists(path):
            print(f"skip {mat}: {path} not found")
            continue
        d = json.load(open(path))
        C = d.get("Chom_sym", d.get("Chom"))
        c = constants_from_C(C)
        E, G, nu = c["E"], c["G"], c["nu"]
        print(f"\n== {mat} (frame {AXES}) ==")
        print(f"  E_x={E[0]:.1f}  E_y={E[1]:.1f}  E_z={E[2]:.1f} GPa")
        print(f"  nu_xy={nu[(0,1)]:.3f} nu_yz={nu[(1,2)]:.3f} nu_xz={nu[(0,2)]:.3f}"
              f"  (nu_yx={nu[(1,0)]:.3f} nu_zy={nu[(2,1)]:.3f} nu_zx={nu[(2,0)]:.3f})")
        print(f"  G_yz={G[0]:.1f}  G_xz={G[1]:.1f}  G_xy={G[2]:.1f} GPa")
        print(f"  residual normal-shear coupling max|C[0:3,3:6]| = {c['coupling']:.1f} GPa")
        rows.append({
            "material": mat,
            "E_x_GPa": round(E[0], 1), "E_y_GPa": round(E[1], 1), "E_z_GPa": round(E[2], 1),
            "nu_xy": round(nu[(0, 1)], 3), "nu_xz": round(nu[(0, 2)], 3),
            "nu_yz": round(nu[(1, 2)], 3), "nu_yx": round(nu[(1, 0)], 3),
            "nu_zx": round(nu[(2, 0)], 3), "nu_zy": round(nu[(2, 1)], 3),
            "G_yz_GPa": round(G[0], 1), "G_xz_GPa": round(G[1], 1), "G_xy_GPa": round(G[2], 1),
            "coupling_GPa": round(c["coupling"], 1),
        })

    if not rows:
        return
    # csv
    with open(os.path.join(args.dir, "engineering_constants.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # markdown table for the report
    md = os.path.join(args.dir, "report", "engineering_constants.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w") as f:
        f.write("| Stahl | E_x (Schweiß) | E_y (Wandn.) | E_z (Aufbau) | ν_xy | ν_xz | ν_yz | G_yz | G_xz | G_xy |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write("| {material} | {E_x_GPa} | {E_y_GPa} | {E_z_GPa} | {nu_xy} | {nu_xz} | "
                    "{nu_yz} | {G_yz_GPa} | {G_xz_GPa} | {G_xy_GPa} |\n".format(**r))
    print(f"\nwrote {md} and engineering_constants.csv")


if __name__ == "__main__":
    main()
