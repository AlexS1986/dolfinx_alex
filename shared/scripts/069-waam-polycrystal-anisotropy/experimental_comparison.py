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

Numerical values are read from the FE uniaxial results Emodul_<mat>_<orient>.json.

Output:
  report/fig_exp_vs_num.png            grouped bar chart (exp vs num), per steel
  report/experimental_comparison.md    markdown table for the report
"""
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


def main():
    num = {m: {o: num_E(m, o) for o in ORIENTS} for m in EXP}

    # ---- figure: grouped bars exp vs num, one panel per steel -------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(ORIENTS)); w = 0.38
    for ax, mat in zip(axs, ["316L", "17-4PH"]):
        e = [EXP[mat][o][0] for o in ORIENTS]; es = [EXP[mat][o][1] for o in ORIENTS]
        n = [num[mat][o] for o in ORIENTS]
        ax.bar(x - w/2, e, w, yerr=es, capsize=4, color="#4c72b0", label="Experiment")
        ax.bar(x + w/2, n, w, color="#c44e52", label="Numerisch (FE)")
        for i, v in enumerate(e): ax.text(i - w/2, v + 4, f"{v:.0f}", ha="center", fontsize=8)
        for i, v in enumerate(n): ax.text(i + w/2, v + 4, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([OLAB[o] for o in ORIENTS])
        ax.set_ylabel("E [GPa]"); ax.set_title(mat); ax.set_ylim(0, 260)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "report", "fig_exp_vs_num.png"), dpi=150, bbox_inches="tight")
    print("wrote report/fig_exp_vs_num.png")

    # ---- markdown table ---------------------------------------------------
    lines = ["| Werkstoff | Orientierung | E Experiment [GPa] | E Numerisch [GPa] | Abw. |",
             "|---|---|---|---|---|"]
    for mat in ["316L", "17-4PH"]:
        for o in ORIENTS:
            e = EXP[mat][o][0]; s = EXP[mat][o][1]; n = num[mat][o]
            d = (n - e) / e * 100
            lines.append(f"| {mat} | {OLAB[o]} | {e:.0f} ± {s:.0f} | {n:.0f} | {d:+.0f} % |")
    md = "\n".join(lines)
    open(os.path.join(HERE, "report", "experimental_comparison.md"), "w").write(md + "\n")
    print("wrote report/experimental_comparison.md\n")
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
