#!/usr/bin/env python3
"""
Post-processing for the KLAIM 2025 revision-2 reruns (2D folder 068 and 3D folder 069).

Reads results/<case>/{parameters.txt, run_simulation_graphs.txt, run_simulation_graphs_columns.txt}
(the layout produced by 05_prepare_lightweight_results_on_cluster.sh + 06_download_results_lightweight.sh)
and writes tables (csv) and figures (png) into ./evaluation_output/.

Usage:
    python3 evaluation_rev2.py                        # results in ./results
    python3 evaluation_rev2.py --results path/to/results --out evaluation_output
    python3 evaluation_rev2.py --legacy path/to/KLAIM_2025_research_data/scripts   # adds old-vs-new overlays

Legacy tables (old column layout) are recognised automatically when no *_graphs_columns.txt file exists.
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LEGACY_2D_VM = ["t", "Jx", "Jy", "x_ct", "x_bc", "Rx_top", "Ry_top", "dW", "W", "A", "dt", "E_el", "E_total", "E_pl"]
LEGACY_2D_EL = ["t", "Jx", "Jy", "x_ct", "x_bc", "Rx_top", "Ry_top", "dW", "W", "A", "dt", "E_el"]
LEGACY_3D = ["t", "Jx", "Jy", "Jz", "x_ct", "x_bc", "Rx_top", "Ry_top", "Rz_top", "dW", "W", "A"]

SET_STYLE = {"vM": dict(color="#1b4f72", marker="o", label="[vM]"),
             "EqJc": dict(color="#b03a2e", marker="s", label="[Eq $J_c$]"),
             "EqSig": dict(color="#1e8449", marker="^", label="[Eq $\\sigma^*$]"),
             "3D_vM": dict(color="#6c3483", marker="D", label="[vM] 3D")}

# Replacement runs (decision 2026-10-21): the S1/S3 originals at w_s = 4.0 were cancelled before the evaluation window;
# their S7 copies with NEWTON_MIN_ITERS = 5 (identical J_eff at w_s = 1.0, CLAUDE.md 2026-10-19) take their place.
BASELINE_REPLACEMENTS = {"S7_vM_ws4.0_mi5": "S1_vM_ws4.0", "S7_cubic_vM_ws4.0_mi5": "S3_cubic_vM_ws4.0"}
SUPERSEDED = set(BASELINE_REPLACEMENTS.values()) | {"S7_vM_ws4.0_mi6"}
# J_c^num of the v2.0 paper (homogeneous strip, beta = 0.1; [Eq sigma*] with J_c = 0.5679) for the old-vs-new comparison
LEGACY_JC_NUM = {"vM": 1.50, "EqJc": 1.12, "EqSig": 0.64}
EQSIG_GC = 0.5523   # [Eq sigma*] J_c of the reruns -> exact self-similarity predicts J_eff(EqJc)/J_eff(EqSig) = 1/0.5523


# ----------------------------------------------------------------------------------------------- reading
def read_parameters(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            try:
                params[k] = int(v)
            except ValueError:
                try:
                    params[k] = float(v)
                except ValueError:
                    params[k] = v
    return params


class Case:
    def __init__(self, folder):
        self.folder = folder
        self.name = os.path.basename(os.path.normpath(folder))
        self.params = read_parameters(os.path.join(folder, "parameters.txt"))
        self.data = np.loadtxt(os.path.join(folder, "run_simulation_graphs.txt"), comments="#", ndmin=2)
        cols_file = os.path.join(folder, "run_simulation_graphs_columns.txt")
        if os.path.exists(cols_file):
            self.columns = [tok.split(":", 1)[1] for tok in open(cols_file).read().split()]
            self.legacy = False
        else:  # legacy layouts
            n = self.data.shape[1]
            self.columns = {14: LEGACY_2D_VM, 12: LEGACY_2D_EL}.get(n, LEGACY_3D if n == 12 else None)
            if self.columns is None:
                raise ValueError(f"{folder}: unknown legacy layout with {n} columns")
            self.legacy = True
        self.dim = 3 if "Jz" in self.columns else 2
        # 3D legacy: 12 columns is ambiguous (elastic 2D vs 3D) -> 3D parameter files use "Nholes"
        if self.legacy and self.data.shape[1] == 12 and "Nholes" in self.params:
            self.columns = LEGACY_3D
            self.dim = 3
        self.col = {c: i for i, c in enumerate(self.columns)}
        self.wsteg = float(self.params.get("wsteg"))
        self.dhole = float(self.params.get("dhole", 1.0))
        self.nholes = int(self.params.get("nholes", self.params.get("Nholes", 6)))
        self.w_cell = self.wsteg + self.dhole
        self.material = self.params.get("material", "vonmises" if "E_pl" in self.columns else "elastic")
        self.gc = float(self.params.get("Gc_simulation", 1.0))
        self.geometry = self.params.get("geometry", "porous")
        self.embedding = self.params.get("embedding", "isotropic")
        self.nrows = int(self.params.get("nrows", 3))
        self.sig_y = float(self.params.get("sig_y_simulation", 1.0))
        self.hard = float(self.params.get("hard_simulation", 0.2222222))
        self.crack_factor = float(self.params.get("in_crack_length", self.w_cell)) / self.w_cell
        # x offset: 3D meshes are centred at the origin (length (nholes + 2) w_cell), 2D meshes start at x = 0; pore_positions()
        # uses the 2D convention (one embedding cell, then the pores), so 3D x values are shifted by half the domain length
        self.x_offset = 0.5 * (self.nholes + 2) * self.w_cell if self.dim == 3 else 0.0
        # 3D mesh: fixed band element size h_band (S1F/S2F/S9F, decision 2026-10-20) or the old w_s-scaled mesh (0.0); 2D: -1.0
        hb = self.params.get("h_band", 0.0)
        self.h_band = (float(hb) if isinstance(hb, (int, float)) else 0.0) if self.dim == 3 else -1.0   # "old_rule" -> 0.0

    def c(self, name):
        return self.data[:, self.col[name]]

    def x_ct(self):
        return self.c("x_ct") + self.x_offset

    @property
    def set_name(self):
        if self.dim == 3:
            return "3D_vM" if self.material == "vonmises" else ("3D_EqSig" if self.gc < 0.9 else "3D_EqJc")
        if self.material == "vonmises":
            return "vM"
        return "EqSig" if self.gc < 0.9 else "EqJc"

    def is_baseline(self):
        if self.name in SUPERSEDED:
            return False
        if self.name.startswith("scaling_") or (self.name not in BASELINE_REPLACEMENTS
                                                and any(tag in self.name for tag in ("_mi", "_nl", "_nh"))):   # S7/S8 variants are not baseline
            return False
        plastic_ok = self.material != "vonmises" or (abs(self.sig_y - 1.0) < 1e-9 and abs(self.hard - 0.2222222) < 1e-6)
        if self.dim == 3:   # paper uses only the fixed-mesh runs with h_band = 0.1 (S1F); S9F (h 0.07) is the convergence check
            return self.geometry == "porous" and self.embedding == "isotropic" and abs(self.h_band - 0.1) < 1e-9 and plastic_ok
        return (self.geometry == "porous" and self.embedding == "isotropic" and self.nrows == 3 and abs(self.crack_factor - 1.0) < 1e-6
                and plastic_ok)


def load_cases(results_dir):
    cases = []
    for folder in sorted(glob.glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(folder) or os.path.basename(folder).startswith(("scaling_", "blas_check", "homtest")):
            continue
        if not (os.path.exists(os.path.join(folder, "parameters.txt")) and os.path.exists(os.path.join(folder, "run_simulation_graphs.txt"))):
            continue
        try:
            cases.append(Case(folder))
        except Exception as e:
            print(f"skipping {folder}: {e}")
    return cases


# ----------------------------------------------------------------------------------------------- J_eff
def pore_positions(case):
    starts, ends = [], []
    for k in range(case.nholes):
        xc = case.w_cell * 1.5 + k * case.w_cell
        starts.append(xc - case.dhole / 2.0)
        ends.append(xc + case.dhole / 2.0)
    return starts, ends


def Jeff(case, starting_hole=None):
    """effective crack resistance (decision 2026-10-23): maximum of J_x over the pore-to-pore cycle `starting_hole` (crack entering
    pore k to entering pore k+1, see energy_cycles). In 2D this is identical to the v2.0 window definition (Jeff_window, checked for
    every case); in 3D the front passes beside the spherical pore and the peak can occur while x_ct is inside the pore's x-range,
    which the window misses for small w_s. Falls back to the window if the cycle is incomplete."""
    if starting_hole is None:
        starting_hole = 1 if case.dim == 3 else 2
    cyc = [d for d in energy_cycles(case) if d["cycle"] == starting_hole]
    return cyc[0]["Jmax"] if cyc else Jeff_window(case, starting_hole)


def Jeff_window(case, starting_hole=None):
    """v2.0 definition: max J_x while the crack tip is in the ligament in front of pore (starting_hole+1)"""
    if starting_hole is None:
        starting_hole = 1 if case.dim == 3 else 2
    starts, _ = pore_positions(case)
    x_high = starts[starting_hole + 1]
    x = case.x_ct()
    mask = (x >= x_high - 1.01 * case.wsteg) & (x <= x_high + 0.01 * case.wsteg)
    if not np.any(mask):
        return float("nan")
    return float(np.max(case.c("Jx")[mask]))


def Jc_num_from_strip(case, return_coverage=False):
    """plateau value of J_x in the homogeneous strip/solid (mean over the central part of the crack path).
    2D: x_ct in (0.3, 0.7) max(x_bc). 3D: x_ct in (0.3, 0.7) L with L = domain length (same as solid_reference_plateau.py;
    x_bc is in centred coordinates there). coverage = fraction of the plateau window already passed by the crack (< 1: running)."""
    x = case.x_ct()
    if case.dim == 3:
        L = 2.0 * case.x_offset
        lo, hi = 0.3 * L, 0.7 * L
    else:
        xmax = np.max(case.c("x_bc"))
        lo, hi = 0.3 * xmax, 0.7 * xmax
    # x-weighted mean (equal weight per crack length): a row mean over-weights slow phases (3D S2F ws1.0: 1.387 vs 1.415)
    x_run = np.maximum.accumulate(x)
    coverage = float(np.clip((x_run[-1] - lo) / (hi - lo), 0.0, 1.0))
    if coverage <= 0.0:
        val = float("nan")
    else:
        xs = np.linspace(lo, lo + coverage * (hi - lo), 400)
        idx = np.clip(np.searchsorted(x_run, xs, side="right") - 1, 0, len(x) - 1)
        val = float(np.mean(case.c("Jx")[idx]))
    return (val, coverage) if return_coverage else val


def first_bin_J(case, width=0.1):
    """mean J_x in the first 0.1 L bin after 0.3 L (3D solids that have not reached the plateau yet, cf. CLAUDE.md 2026-10-20)"""
    x = case.x_ct()
    L = 2.0 * case.x_offset if case.dim == 3 else np.max(case.c("x_bc"))
    mask = (x > 0.3 * L) & (x < (0.3 + width) * L)
    return float(np.mean(case.c("Jx")[mask])) if np.any(mask) else float("nan")


def solid_reference_3d(case, solids):
    """J_c^num for a 3D porous run: solid on the same mesh (h_band) and w_s, else same h_band and nearest w_s; never across meshes"""
    same_mesh = [s for s in solids if abs(s["h_band"] - case.h_band) < 1e-9 and s["h_band"] > 0]
    if not same_mesh:
        return None
    return min(same_mesh, key=lambda s: (abs(s["wsteg"] - case.wsteg), -s["coverage"]))


# ----------------------------------------------------------------------------------------------- energies
def reference_index(case):
    """first index after the initial relaxation of s (minimum of the regularised crack surface A)"""
    return int(np.argmin(case.c("A")))


def energy_partition(case):
    """returns dict of cumulative energies relative to the reference state (per unit thickness)"""
    i0 = reference_index(case)
    out = {"t": case.c("t")[i0:], "x_ct": case.x_ct()[i0:], "Jx": case.c("Jx")[i0:]}
    out["W"] = case.c("W")[i0:] - case.c("W")[i0]
    if "E_el" not in case.col:  # legacy 3D tables: no energies recorded
        nan = np.full_like(out["W"], np.nan)
        out.update({"E_el": nan, "E_frac": case.gc * (case.c("A")[i0:] - case.c("A")[i0]), "E_pl": nan, "D_visc": nan, "D_pl_deg": nan, "residual": nan})
        return out
    out["E_el"] = case.c("E_el")[i0:] - case.c("E_el")[i0]
    if "E_frac" in case.col:
        out["E_frac"] = case.c("E_frac")[i0:] - case.c("E_frac")[i0]
    else:
        out["E_frac"] = case.gc * (case.c("A")[i0:] - case.c("A")[i0])
    out["E_pl"] = (case.c("E_pl")[i0:] - case.c("E_pl")[i0]) if "E_pl" in case.col else np.zeros_like(out["W"])
    if "D_visc" in case.col:
        out["D_visc"] = case.c("D_visc")[i0:] - case.c("D_visc")[i0]
        out["D_pl_deg"] = case.c("D_pl_deg")[i0:] - case.c("D_pl_deg")[i0]
        out["residual"] = out["W"] - out["E_el"] - out["E_frac"] - out["E_pl"] - out["D_visc"]
        out["E_in"] = out["E_el"] + out["E_frac"] + out["E_pl"] + out["D_visc"]   # supplied energy from the sinks (W column unreliable)
    else:
        out["D_visc"] = out["W"] - out["E_el"] - out["E_frac"] - out["E_pl"]  # = unrecorded dissipation (legacy)
        out["D_pl_deg"] = np.full_like(out["W"], np.nan)
        out["residual"] = np.zeros_like(out["W"])
        out["E_in"] = out["W"].copy()
    return out


def plot_energy_partition(case, out_dir):
    e = energy_partition(case)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(e["x_ct"], e["W"], color="k", lw=1.6, label="$\\Delta W$ external work")
    ax.plot(e["x_ct"], e["E_el"], color="#e67e22", label="$\\Delta E_{el}$")
    ax.plot(e["x_ct"], e["E_frac"], color="#1e8449", label="$\\Delta E_{fr}$ fracture")
    if np.any(np.nan_to_num(e["E_pl"]) != 0):
        ax.plot(e["x_ct"], e["E_pl"], color="#b03a2e", label="$\\Delta E_{pl}$ (stored, degraded)")
    ax.plot(e["x_ct"], e["D_visc"], color="#6c3483", ls="--", label="$D_{visc}$ (viscous, crack jumps)" if not case.legacy else "unrecorded dissipation")
    if not case.legacy and np.any(np.isfinite(e["D_pl_deg"])):
        ax.plot(e["x_ct"], e["D_pl_deg"], color="#b03a2e", ls=":", label="plastic work $\\int g\\,\\sigma_Y\\dot\\alpha$")
        ax.plot(e["x_ct"], e["residual"], color="gray", ls=":", lw=0.8, label="balance residual")
    for xs in pore_positions(case)[0]:
        ax.axvline(xs, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("$x_{ct}/L$")
    ax.set_ylabel("energy / ($J_c^0 L$)")
    ax.set_title(f"{case.name}: $w_s$={case.wsteg}, {case.material}, $J_c$={case.gc}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"energy_partition_{case.name}.png"), dpi=150)
    plt.close(fig)


def dtot(e, i):
    """dissipated energy (fracture + stored plastic + viscous) at index i, relative to the reference state"""
    return e["E_frac"][i] + e["E_pl"][i] + e["D_visc"][i]


def energy_cycles(case):
    """energy partition per pore-to-pore cycle: cycle k runs from the crack entering pore k (running max of x_ct passes the pore
    centre) to the crack entering pore k+1, i.e. loading, re-nucleation and failure of the ligament between pores k and k+1.
    The energy of a crack jump is released over many small steps around the jump, so a cycle is more robust than a single step.
    The J_eff ligament lies in cycle `starting_hole` (2D: 2, 3D: 1)."""
    e = energy_partition(case)
    starts, ends = pore_positions(case)
    centres = 0.5 * (np.asarray(starts) + np.asarray(ends))
    x_run = np.maximum.accumulate(e["x_ct"])   # x_ct flickers by single nodes around jumps
    idx = [int(np.argmax(x_run >= xc)) if np.any(x_run >= xc) else None for xc in centres]
    cycles = []
    for k in range(len(centres) - 1):
        a, b = idx[k], idx[k + 1]
        if a is None or b is None:
            continue
        d = {q: float(e[q][b] - e[q][a]) for q in ("E_el", "E_frac", "E_pl", "D_visc", "E_in")}
        d.update(cycle=k, x_from=float(x_run[a]), x_to=float(x_run[b]), t_from=float(e["t"][a]), t_to=float(e["t"][b]),
                 Jmax=float(np.max(e["Jx"][a:b + 1])))
        d["D_tot"] = d["E_frac"] + d["E_pl"] + d["D_visc"]
        d["G_diss"] = d["D_tot"] / case.w_cell   # dissipated (incl. stored plastic) energy per unit crack advance
        cycles.append(d)
    return cycles


def eval_cycle_fields(case):
    """partition of the cycle that contains the J_eff ligament (fractions of the energy supplied in that cycle)"""
    k_eval = 1 if case.dim == 3 else 2
    cyc = [d for d in energy_cycles(case) if d["cycle"] == k_eval]
    if not cyc:
        return dict(cyc_E_in=np.nan, cyc_el=np.nan, cyc_fracture=np.nan, cyc_plastic=np.nan, cyc_visc=np.nan, cyc_G_diss=np.nan)
    d = cyc[0]
    return dict(cyc_E_in=d["E_in"], cyc_el=d["E_el"] / d["D_tot"], cyc_fracture=d["E_frac"] / d["D_tot"],
                cyc_plastic=d["E_pl"] / d["D_tot"], cyc_visc=d["D_visc"] / d["D_tot"], cyc_G_diss=d["G_diss"])


def event_energies(case):
    """energy released/consumed per crack jump (re-nucleation event): list of dicts"""
    e = energy_partition(case)
    x = e["x_ct"]
    jumps = np.where(np.diff(x) > 4.0 * float(case.params.get("eps", 0.1)))[0]
    events = []
    for j in jumps:
        d = {k: float(e[k][j + 1] - e[k][j]) for k in ("W", "E_el", "E_frac", "E_pl", "D_visc")}
        d["x_before"], d["x_after"], d["Jx_before"] = float(x[j]), float(x[j + 1]), float(e["Jx"][j])
        events.append(d)
    return events


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    ap.add_argument("--results3d", default=None, help="results folder of the 3D project (069), optional")
    ap.add_argument("--legacy", default=None, help="path to KLAIM_2025_research_data/scripts for old-vs-new overlays")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation_output"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cases = load_cases(args.results)
    if args.results3d:
        cases += load_cases(args.results3d)
    if not cases:
        print("no cases found in", args.results)
        sys.exit(1)
    dropped = sorted(c.name for c in cases if c.name in SUPERSEDED)
    cases = [c for c in cases if c.name not in SUPERSEDED]
    print(f"loaded {len(cases)} cases (superseded, not evaluated: {', '.join(dropped) or '-'})")

    # --- J_c^num from strip runs
    jc_num = {}       # from the "solid" run (one homogeneous material) when present, else from the "strip" run (elastic embedding)
    jc_num_all = {}
    solids3d = []     # 3D: one reference per (mesh, w_s), assigned per porous run in solid_reference_3d()
    for c in cases:
        if c.geometry in ("strip", "solid") and c.dim == 3:
            v, cov = Jc_num_from_strip(c, return_coverage=True)
            solids3d.append(dict(name=c.name, h_band=c.h_band, wsteg=c.wsteg, Jc_num=v, coverage=cov, J_first_bin=first_bin_J(c)))
        elif c.geometry in ("strip", "solid"):
            jc_num_all[(c.set_name, c.geometry)] = Jc_num_from_strip(c)
    for (name, geo), v in jc_num_all.items():
        if geo == "solid" or name not in jc_num:
            jc_num[name] = v
    print("J_c^num from strip/solid runs:", jc_num_all, "-> used:", jc_num)
    with open(os.path.join(args.out, "Jc_num_strip.csv"), "w") as f:
        f.write("set,geometry,Jc_num,used\n")
        for (name, geo), v in jc_num_all.items():
            f.write(f"{name},{geo},{v:.5f},{int(jc_num[name] == v)}\n")
    with open(os.path.join(args.out, "Jc_num_solid_3D.csv"), "w") as f:
        f.write("case,h_band,wsteg,Jc_num_plateau,plateau_coverage,J_first_bin_0.3_0.4L\n")
        for s in sorted(solids3d, key=lambda s: (s["h_band"], s["wsteg"])):
            f.write(f"{s['name']},{s['h_band']},{s['wsteg']},{s['Jc_num']:.4f},{s['coverage']:.3f},{s['J_first_bin']:.4f}\n")
            print(f"3D solid {s['name']}: h_band {s['h_band']} w_s {s['wsteg']} plateau {s['Jc_num']:.4f} (coverage {s['coverage']:.2f}), first bin {s['J_first_bin']:.4f}")

    def jc_ref(c):
        """(J_c^num, reference name, coverage) used to normalise case c"""
        if c.dim == 3:
            s = solid_reference_3d(c, solids3d)
            return (s["Jc_num"], s["name"], s["coverage"]) if s else (np.nan, "", np.nan)
        return (jc_num.get(c.set_name, np.nan), c.set_name, 1.0)

    # --- master table
    rows = []
    for c in cases:
        if c.geometry in ("strip", "solid"):
            continue
        je = Jeff(c)
        e = energy_partition(c)
        rows.append(dict(name=c.name, set=c.set_name, dim=c.dim, wsteg=c.wsteg, material=c.material, gc=c.gc, sig_y=c.sig_y, hard=c.hard,
                         embedding=c.embedding, nrows=c.nrows, crack_factor=round(c.crack_factor, 3), baseline=c.is_baseline(),
                         h_band=c.h_band, Jeff=je, Jeff_over_Jcnum=je / jc_ref(c)[0], Jc_ref=jc_ref(c)[1], Jc_ref_value=jc_ref(c)[0],
                         Jc_ref_coverage=jc_ref(c)[2],
                         W_end=e["W"][-1], E_frac_end=e["E_frac"][-1], E_pl_end=e["E_pl"][-1], D_visc_end=e["D_visc"][-1],
                         frac_fracture=e["E_frac"][-1] / e["W"][-1], frac_visc=e["D_visc"][-1] / e["W"][-1], frac_plastic=e["E_pl"][-1] / e["W"][-1],
                         residual_end_rel=e["residual"][-1] / e["W"][-1], residual_maxabs_rel=np.max(np.abs(e["residual"])) / e["W"][-1],
                         # partition of the dissipated energy D_tot = dE_frac + dE_pl + D_visc (fractions sum to 1); el = dE_el / D_tot.
                         # The tabulated W integrates the traction over ALL exterior facets incl. the pore walls and is not used
                         # (test 2026-10-21, CLAUDE.md); E_in = sum of the sinks is the energy actually supplied.
                         E_in_end=e["E_in"][-1], D_tot_end=dtot(e, -1), sink_el=e["E_el"][-1] / dtot(e, -1),
                         sink_fracture=e["E_frac"][-1] / dtot(e, -1), sink_plastic=e["E_pl"][-1] / dtot(e, -1),
                         sink_visc=e["D_visc"][-1] / dtot(e, -1),
                         **eval_cycle_fields(c),
                         zener=c.params.get("zener_effective", np.nan), C11=c.params.get("C11_effective", np.nan),
                         C12=c.params.get("C12_effective", np.nan), C33_or_C44=c.params.get("C33_effective", c.params.get("C44_effective", np.nan)),
                         lam_eff=c.params.get("lam_effective", np.nan), mu_eff=c.params.get("mue_effective", np.nan),
                         n_events=len(event_energies(c)), t_end=c.c("t")[-1], n_steps=len(c.c("t"))))
    keys = list(rows[0].keys())
    with open(os.path.join(args.out, "master_table.csv"), "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=keys)
        wri.writeheader()
        wri.writerows(rows)

    # --- J_eff vs w_s (baseline sets + 3D), raw and normalised
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sname in ("vM", "EqJc", "EqSig", "3D_vM"):
        sel = sorted([r for r in rows if r["set"] == sname and r["baseline"]], key=lambda r: r["wsteg"])
        if not sel:
            continue
        st = SET_STYLE.get(sname, dict(marker="x", label=sname))
        axes[0].plot([r["wsteg"] for r in sel], [r["Jeff"] for r in sel], **st)
        axes[1].plot([r["wsteg"] for r in sel], [r["Jeff_over_Jcnum"] for r in sel], **st)
    axes[0].set_ylabel("$J_{x,c}^{eff} / J_c^0$"); axes[1].set_ylabel("$J_{x,c}^{eff} / J_c^{num}$ (own strip value)")
    for ax in axes:
        ax.set_xlabel("$w_s / L$"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "Jeff_vs_wsteg.png"), dpi=150); plt.close(fig)

    # --- energy fractions vs w_s
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for sname in ("vM", "EqJc", "EqSig"):
        sel = sorted([r for r in rows if r["set"] == sname and r["baseline"]], key=lambda r: r["wsteg"])
        if not sel:
            continue
        col = SET_STYLE[sname]["color"]
        ws = [r["wsteg"] for r in sel]
        ax.plot(ws, [r["sink_fracture"] for r in sel], color=col, marker="o", label=f"{SET_STYLE[sname]['label']} fracture")
        ax.plot(ws, [r["sink_visc"] for r in sel], color=col, marker="s", ls="--", label=f"{SET_STYLE[sname]['label']} viscous (jumps)")
        if sname == "vM":
            ax.plot(ws, [r["sink_plastic"] for r in sel], color=col, marker="^", ls=":", label="[vM] plastic (stored)")
    ax.set_xlabel("$w_s / L$"); ax.set_ylabel("fraction of dissipated energy (whole run)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "energy_fractions_vs_wsteg.png"), dpi=150); plt.close(fig)

    # --- per-case energy partition plots, event tables, ligament stress plots
    with open(os.path.join(args.out, "events.csv"), "w") as f:
        f.write("case,set,wsteg,x_before,x_after,Jx_before,dW,dE_el,dE_frac,dE_pl,dD_visc\n")
        for c in cases:
            if c.geometry in ("strip", "solid"):
                continue
            plot_energy_partition(c, args.out)
            for ev in event_energies(c):
                f.write(f"{c.name},{c.set_name},{c.wsteg},{ev['x_before']:.4f},{ev['x_after']:.4f},{ev['Jx_before']:.4f},"
                        f"{ev['W']:.4f},{ev['E_el']:.4f},{ev['E_frac']:.4f},{ev['E_pl']:.4f},{ev['D_visc']:.4f}\n")
            if "sig_vM_max_lig" in c.col:
                fig, ax = plt.subplots(figsize=(7, 4.0))
                x = c.x_ct()
                ax.plot(x, c.c("sig_vM_max_lig") / c.sig_y if c.material == "vonmises" else c.c("sig_vM_max_lig"), label="max $\\sigma_{vM}$ in next ligament" + (" / $\\sigma_y$" if c.material == "vonmises" else ""))
                ax.plot(x, c.c("sig_h_max_lig"), label="max $\\sigma_h$ in next ligament")
                ax.plot(x, c.c("sig_yy_max_lig"), label="max $\\sigma_{yy}$ in next ligament")
                ax2 = ax.twinx()
                ax2.plot(x, c.c("Jx"), color="k", lw=0.8, alpha=0.6, label="$J_x$")
                if c.material == "vonmises":
                    ax2.plot(x, c.c("alpha_max_lig"), color="gray", lw=0.8, ls="--", label="max $\\alpha$ in next ligament")
                for xs in pore_positions(c)[0]:
                    ax.axvline(xs, color="0.85", lw=0.8, zorder=0)
                ax.set_xlabel("$x_{ct}/L$"); ax.set_ylabel("stress"); ax2.set_ylabel("$J_x$, $\\alpha$")
                ax.set_title(f"{c.name}"); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper left"); ax2.legend(fontsize=7, loc="upper right")
                fig.tight_layout(); fig.savefig(os.path.join(args.out, f"ligament_stress_{c.name}.png"), dpi=150); plt.close(fig)

    # --- energy partition per pore-to-pore cycle (all cases) and for the evaluation cycle vs w_s
    with open(os.path.join(args.out, "energy_cycles.csv"), "w") as f:
        f.write("case,set,dim,wsteg,baseline,cycle,x_from,x_to,t_from,t_to,Jmax,E_in,dE_el,dE_frac,dE_pl,dD_visc,"
                "el_over_Dtot,frac_fracture,frac_plastic,frac_visc,G_diss\n")   # fractions of D_tot
        for c in cases:
            if c.geometry in ("strip", "solid"):
                continue
            for d in energy_cycles(c):
                f.write(f"{c.name},{c.set_name},{c.dim},{c.wsteg},{int(c.is_baseline())},{d['cycle']},{d['x_from']:.4f},{d['x_to']:.4f},"
                        f"{d['t_from']:.5f},{d['t_to']:.5f},{d['Jmax']:.4f},{d['E_in']:.5f},{d['E_el']:.5f},{d['E_frac']:.5f},{d['E_pl']:.5f},"
                        f"{d['D_visc']:.5f},{d['E_el'] / d['D_tot']:.4f},{d['E_frac'] / d['D_tot']:.4f},{d['E_pl'] / d['D_tot']:.4f},"
                        f"{d['D_visc'] / d['D_tot']:.4f},{d['G_diss']:.4f}\n")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for sname in ("vM", "EqJc", "EqSig"):
        sel = sorted([r for r in rows if r["set"] == sname and r["baseline"] and r["dim"] == 2 and np.isfinite(r["cyc_E_in"])],
                     key=lambda r: r["wsteg"])
        if not sel:
            continue
        col, lab = SET_STYLE[sname]["color"], SET_STYLE[sname]["label"]
        ws = [r["wsteg"] for r in sel]
        axes[0].plot(ws, [r["cyc_fracture"] for r in sel], color=col, marker="o", label=f"{lab} fracture")
        axes[0].plot(ws, [r["cyc_visc"] for r in sel], color=col, marker="s", ls="--", label=f"{lab} viscous (jumps)")
        if sname == "vM":
            axes[0].plot(ws, [r["cyc_plastic"] for r in sel], color=col, marker="^", ls=":", label=f"{lab} plastic (stored)")
        axes[1].plot(ws, [r["cyc_G_diss"] for r in sel], color=col, marker="o", label=f"{lab} $G_{{diss}}$ (cycle)")
        axes[1].plot(ws, [r["Jeff"] for r in sel], color=col, marker="x", ls=":", label=f"{lab} $J_{{x,c}}^{{eff}}$")
    axes[0].set_ylabel("fraction of the energy dissipated in the cycle"); axes[1].set_ylabel("$/(J_c^0)$")
    axes[0].set_title("evaluation cycle (ligament of $J_{x,c}^{eff}$)", fontsize=9)
    axes[1].set_title("$G_{diss}$ = (fracture + plastic + viscous) / $w_c$", fontsize=9)
    for ax in axes:
        ax.set_xlabel("$w_s / L$"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "energy_cycle_partition_vs_wsteg.png"), dpi=150); plt.close(fig)

    # --- cubic vs isotropic, sensitivity, representativity tables
    def find(rows_, **kw):
        return [r for r in rows_ if all(abs(r[k] - v) < 1e-6 if isinstance(v, float) else r[k] == v for k, v in kw.items())]

    with open(os.path.join(args.out, "cubic_vs_isotropic.csv"), "w") as f:
        f.write("dim,wsteg,Jeff_isotropic,Jeff_cubic,rel_change_percent,zener\n")
        for r in [r for r in rows if r["embedding"] == "cubic"]:
            ref = find(rows, set=r["set"], wsteg=r["wsteg"], embedding="isotropic", h_band=r["h_band"],
                       baseline=r["dim"] == 2 or abs(r["h_band"] - 0.1) < 1e-9)   # same mesh only (3D old mesh: no fixed-mesh twin)
            if ref:
                f.write(f"{r['dim']},{r['wsteg']},{ref[0]['Jeff']:.4f},{r['Jeff']:.4f},{100 * (r['Jeff'] / ref[0]['Jeff'] - 1):.2f},{r['zener']}\n")

    sens = [r for r in rows if r["set"] == "vM" and r["dim"] == 2 and r["embedding"] == "isotropic" and r["nrows"] == 3 and abs(r["crack_factor"] - 1) < 1e-6
            and (r["name"].startswith("S4_") or (r["baseline"] and r["wsteg"] in {x["wsteg"] for x in rows if x["name"].startswith("S4_")}))]
    with open(os.path.join(args.out, "sensitivity.csv"), "w") as f:
        f.write("wsteg,sig_y,hard,Jeff,Jeff_rel_to_baseline\n")
        for r in sorted(sens, key=lambda r: (r["wsteg"], r["sig_y"], r["hard"])):
            base = find(rows, set="vM", wsteg=r["wsteg"], baseline=True)
            rel = r["Jeff"] / base[0]["Jeff"] if base else np.nan
            f.write(f"{r['wsteg']},{r['sig_y']},{r['hard']},{r['Jeff']:.4f},{rel:.4f}\n")
    if any(not r["baseline"] for r in sens):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ws in sorted(set(r["wsteg"] for r in sens if not r["baseline"])):
            s_sy = sorted([r for r in sens if abs(r["wsteg"] - ws) < 1e-9 and abs(r["hard"] - 0.2222222) < 1e-6], key=lambda r: r["sig_y"])
            s_h = sorted([r for r in sens if abs(r["wsteg"] - ws) < 1e-9 and abs(r["sig_y"] - 1.0) < 1e-9], key=lambda r: r["hard"])
            axes[0].plot([r["sig_y"] for r in s_sy], [r["Jeff"] for r in s_sy], marker="o", label=f"$w_s$={ws}")
            axes[1].plot([r["hard"] for r in s_h], [r["Jeff"] for r in s_h], marker="o", label=f"$w_s$={ws}")
        axes[0].set_xlabel("$\\sigma_y / \\sqrt{J_c^0\\mu_0/L}$"); axes[1].set_xlabel("$H/\\mu_0$")
        for ax in axes:
            ax.set_ylabel("$J_{x,c}^{eff}/J_c^0$"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "sensitivity_sigy_H.png"), dpi=150); plt.close(fig)

    with open(os.path.join(args.out, "representativity.csv"), "w") as f:
        f.write("case,set,wsteg,nrows,crack_factor,Jeff,Jeff_baseline,rel_change_percent\n")
        for r in [r for r in rows if (r["nrows"] != 3 or abs(r["crack_factor"] - 1) > 1e-6)]:
            base = find(rows, set=r["set"], wsteg=r["wsteg"], baseline=True)
            jb = base[0]["Jeff"] if base else np.nan
            f.write(f"{r['name']},{r['set']},{r['wsteg']},{r['nrows']},{r['crack_factor']},{r['Jeff']:.4f},{jb:.4f},{100 * (r['Jeff'] / jb - 1):.2f}\n")

    # --- Zener table
    with open(os.path.join(args.out, "effective_stiffness_table.csv"), "w") as f:
        f.write("dim,wsteg,lam_eff,mu_eff,C11,C12,C33_or_C44,zener,C11_minus_isotropic_percent\n")
        seen = set()
        for r in sorted(rows, key=lambda r: (r["dim"], r["wsteg"])):
            key = (r["dim"], r["wsteg"])
            if key in seen or not np.isfinite(r["zener"] if r["zener"] is not None else np.nan):
                continue
            seen.add(key)
            c11_iso = r["lam_eff"] + 2 * r["mu_eff"]
            f.write(f"{r['dim']},{r['wsteg']},{r['lam_eff']:.5f},{r['mu_eff']:.5f},{r['C11']:.5f},{r['C12']:.5f},{r['C33_or_C44']:.5f},{r['zener']:.4f},{100 * (r['C11'] / c11_iso - 1):.2f}\n")

    # --- J_x(x_ct) overlays for the reference wall width, old vs new (regression check for beta/eta/embedding changes)
    if args.legacy:
        legacy_dirs = {"vM": os.path.join(args.legacy, "046_plasticity", "results", "gc_1.0_elastic"),
                       "EqJc": os.path.join(args.legacy, "045_standard_holes_PAPER", "results", "gc_1.0"),
                       "EqSig": os.path.join(args.legacy, "045_standard_holes_PAPER", "results", "interpolation_as_in_plasticity_gc_0.5679")}
        for sname, ldir in legacy_dirs.items():
            if not os.path.isdir(ldir):
                continue
            old = load_cases(ldir)
            for c in [c for c in cases if c.set_name == sname and c.is_baseline() and c.dim == 2 and c.geometry == "porous"]:
                o = [oc for oc in old if abs(oc.wsteg - c.wsteg) < 1e-9]
                if not o:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(o[0].x_ct(), o[0].c("Jx"), color="gray", lw=0.8, label=f"old (v2.0 data) J_eff={Jeff(o[0]):.3f}")
                ax.plot(c.x_ct(), c.c("Jx"), color=SET_STYLE[sname]["color"], lw=0.8, label=f"new J_eff={Jeff(c):.3f}")
                for xs in pore_positions(c)[0]:
                    ax.axvline(xs, color="0.85", lw=0.8, zorder=0)
                ax.set_xlabel("$x_{ct}/L$"); ax.set_ylabel("$J_x/J_c^0$"); ax.set_title(f"{sname}, $w_s$={c.wsteg}"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
                fig.tight_layout(); fig.savefig(os.path.join(args.out, f"old_vs_new_{sname}_ws{c.wsteg}.png"), dpi=150); plt.close(fig)

    # --- 2D summary: J_eff/J_c^num per set over w_s, v2.0 values, self-similarity and energy-balance checks
    legacy_jeff = {}
    if args.legacy:
        for sname, ldir in legacy_dirs.items():
            if os.path.isdir(ldir):
                for oc in load_cases(ldir):
                    legacy_jeff[(sname, round(oc.wsteg, 4))] = Jeff(oc)
    base2d = [r for r in rows if r["dim"] == 2 and r["baseline"]]
    by = {(r["set"], round(r["wsteg"], 4)): r for r in base2d}
    sets = ("vM", "EqJc", "EqSig")
    hdr = ["wsteg"]
    for s in sets:
        hdr += [f"Jeff_{s}", f"Jeff_over_Jcnum_{s}", f"case_{s}", f"Jeff_v2_{s}", f"Jeff_over_Jcnum_v2_{s}", f"residual_maxabs_rel_{s}"]
    hdr += ["ratio_EqJc_EqSig", "ratio_EqJc_EqSig_target", "ratio_normalised_EqJc_EqSig"]
    with open(os.path.join(args.out, "Jeff_summary_2D.csv"), "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(hdr)
        for ws in sorted({k[1] for k in by}):
            line = [ws]
            for s in sets:
                r = by.get((s, ws))
                jv2 = legacy_jeff.get((s, ws), np.nan)
                line += [r["Jeff"] if r else np.nan, r["Jeff_over_Jcnum"] if r else np.nan, r["name"] if r else "",
                         jv2, jv2 / LEGACY_JC_NUM[s], r["residual_maxabs_rel"] if r else np.nan]
            a, b = by.get(("EqJc", ws)), by.get(("EqSig", ws))
            line += [a["Jeff"] / b["Jeff"] if a and b else np.nan, 1.0 / EQSIG_GC,
                     a["Jeff_over_Jcnum"] / b["Jeff_over_Jcnum"] if a and b else np.nan]
            wri.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in line])

    print("wrote", args.out)


if __name__ == "__main__":
    main()
