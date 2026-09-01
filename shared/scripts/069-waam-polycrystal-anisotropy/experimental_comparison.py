#!/usr/bin/env python3
"""
Compare the numerical directional Young's moduli (uniaxial FE bars) with the
EXPERIMENTAL uniaxial tensile tests (RPTU Kaiserslautern, Lehrstuhl fuer
Werkstoffkunde / Prof. Beck) for both steels in V / H / 45deg orientation.

Experimental values (E in GPa, mean +/- std) as provided:
  316L   (specimens A5): V 102.1+/-6,  45deg 173.4+/-6,  H 92.9+/-2
                          nu: V 0.394, 45deg 0.022, H 0.388
  17-4PH (specimens D8): V 186.4+/-1,  45deg 192.0+/-2,  H 169.1+/-2
                          nu: V 0.303, 45deg 0.239, H 0.301

Numerical values: FE uniaxial bars (Emodul_<mat>_<orient>.json) AND the
directional modulus evaluated from the homogenised tensor (Chom_<mat>.json,
1/E(n) = n n n n : S) in the same three load directions.

Output (in --out, default ./report):
  fig_exp_vs_num.png            grouped bars: Experiment / FE-Zugstab / RVE, per steel
  experimental_comparison.md    markdown table for the report

Usage:  python3 experimental_comparison.py [--out DIR]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --- experimental data (E in GPa, std) -------------------------------------
EXP = {
    "316L":   {"V": (102.1, 6.0), "H": (92.9, 2.0), "45deg": (173.4, 6.0)},
    "17-4PH": {"V": (186.4, 1.0), "H": (169.1, 2.0), "45deg": (192.0, 2.0)},
}
EXP_NU = {
    "316L":   {"V": 0.394, "H": 0.388, "45deg": 0.022},
    "17-4PH": {"V": 0.303, "H": 0.301, "45deg": 0.239},
}
ORIENTS = ["V", "H", "45deg"]
OLAB = {"V": "V (∥ Aufbau)", "H": "H (⊥ Aufbau)", "45deg": "45°"}


def num_E(mat, orient):
    p = os.path.join(HERE, f"Emodul_{mat}_{orient}.json")
    return json.load(open(p))["E_apparent_GPa"]


_VOIGT = {(0, 0): 0, (1, 1): 1, (2, 2): 2, (1, 2): 3, (2, 1): 3,
          (0, 2): 4, (2, 0): 4, (0, 1): 5, (1, 0): 5}
# load direction per specimen orientation in the RVE frame (x=weld, y=wall-normal, z=build)
_DIR = {"V": (0, 0, 1.0), "H": (1.0, 0, 0), "45deg": (1.0, 0, 1.0)}


def rve_E(mat, orient):
    """Directional modulus from Chom (Voigt, engineering shear) along the
    specimen load axis: 1/E(n) = n_i n_j n_k n_l S_ijkl."""
    C = np.array(json.load(open(os.path.join(HERE, f"Chom_{mat}.json")))["Chom_sym"])
    S = np.linalg.inv(C)
    n = np.array(_DIR[orient]); n = n / np.linalg.norm(n)
    inv_e = 0.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    a, b = _VOIGT[(i, j)], _VOIGT[(k, l)]
                    f = (0.5 if a > 2 else 1.0) * (0.5 if b > 2 else 1.0)
                    inv_e += n[i] * n[j] * n[k] * n[l] * S[a, b] * f
    return 1.0 / inv_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "report"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    num = {m: {o: num_E(m, o) for o in ORIENTS} for m in EXP}
    rve = {m: {o: rve_E(m, o) for o in ORIENTS} for m in EXP}

    # ---- figure: grouped bars exp vs num, one panel per steel -------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(ORIENTS)); w = 0.27
    for ax, mat in zip(axs, ["316L", "17-4PH"]):
        e = [EXP[mat][o][0] for o in ORIENTS]; es = [EXP[mat][o][1] for o in ORIENTS]
        n = [num[mat][o] for o in ORIENTS]; r = [rve[mat][o] for o in ORIENTS]
        ax.bar(x - w, e, w, yerr=es, capsize=4, color="#4c72b0", label="Experiment (RPTU/WKK)")
        ax.bar(x, n, w, color="#c44e52", label="FE-Zugstab")
        ax.bar(x + w, r, w, color="#dd8452", label="RVE (KUBC)")
        for i, v in enumerate(e): ax.text(i - w, v + 4, f"{v:.0f}", ha="center", fontsize=8)
        for i, v in enumerate(n): ax.text(i, v + 4, f"{v:.0f}", ha="center", fontsize=8)
        for i, v in enumerate(r): ax.text(i + w, v + 4, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([OLAB[o] for o in ORIENTS])
        ax.set_ylabel("E [GPa]"); ax.set_title(mat); ax.set_ylim(0, 260)
        ax.legend(fontsize=8, loc="upper left"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Richtungsabhängiger E-Modul: Experiment vs. Modell", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fig_exp_vs_num.png"), dpi=150, bbox_inches="tight")
    print("wrote", os.path.join(args.out, "fig_exp_vs_num.png"))

    # ---- markdown table ---------------------------------------------------
    lines = ["| Werkstoff | Orientierung | E Experiment | RVE (KUBC) | Abw. | FE-Zugstab | Abw. |",
             "|---|---|---|---|---|---|---|"]
    for mat in ["316L", "17-4PH"]:
        for o in ORIENTS:
            e = EXP[mat][o][0]; s = EXP[mat][o][1]; n = num[mat][o]; r = rve[mat][o]
            dn = (n - e) / e * 100; dr = (r - e) / e * 100
            lines.append(f"| {mat} | {OLAB[o]} | {e:.0f} ± {s:.0f} | {r:.0f} | {dr:+.0f} % "
                         f"| {n:.0f} | {dn:+.0f} % |")
    md = "\n".join(lines)
    open(os.path.join(args.out, "experimental_comparison.md"), "w").write(md + "\n")
    print("wrote", os.path.join(args.out, "experimental_comparison.md"), "\n")
    print(md)

    # ---- console summary --------------------------------------------------
    print("\n-- single-crystal directional moduli (for interpretation) --")
    for mat in ["316L", "17-4PH"]:
        C = json.load(open(os.path.join(HERE, f"Chom_{mat}.json")))["single_crystal_cubic_GPa"]
        c = C["fcc"] if mat == "316L" else C["bcc"]
        C11, C12, C44 = c["C11"], c["C12"], c["C44"]
        S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
        S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
        S44 = 1.0 / C44
        aniso = S11 - S12 - 0.5 * S44
        E100 = 1 / S11
        E110 = 1 / (S11 - 2 * aniso * 0.25)
        E111 = 1 / (S11 - 2 * aniso / 3.0)
        nu100 = -S12 / S11
        print(f"  {mat}: E<100>={E100:.0f}  E<110>={E110:.0f}  E<111>={E111:.0f} GPa;  nu<100>={nu100:.2f}")


if __name__ == "__main__":
    main()
