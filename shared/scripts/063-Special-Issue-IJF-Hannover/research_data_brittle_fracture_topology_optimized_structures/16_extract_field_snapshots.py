#!/usr/bin/env python3
"""Extract small field snapshots from the campaign 260930_M1e5 result files (XDMF/HDF5) for the manuscript plots.

For every run: mesh (geometry, topology), E and gc, and the fields s and u at
  - t = 0.003 s (elastic reference state, also the stress state sig for the principal-stress plots),
  - the end of the first load drop (stored state closest to the running minimum of the total reaction after the first
    peak, i.e. directly after the first crack event),
  - the end of the simulation,
  - for the crack-evolution snapshots additionally all stored states (only for E_var, beta_phi = 0.01, beta_s = 0.03).
Needs numpy + h5py only (no DOLFINx). Output: one compressed .npz per run in --out.

Usage: python3 16_extract_field_snapshots.py --campaign-root /Volumes/MacbookExtension/063-Special-Issue_IJF-Hannover/campaign_260930_M1e5 --out DIR
"""
import argparse, glob, json, os, re
import numpy as np
import h5py

RE = re.compile(r"results_beta_(?P<beta>0_\d+)_a_6_rho_(?P<rho>0_\d+)_(?:var|min|max)_(?P<case>vary|min|max)_spectral(?:_eps(?P<eps>[0-9_]+))?_M100000\.h5$")
EPS_DIR = re.compile(r"_EPS(\d+_\d+)")


def first_peak(r):
    b = 0
    for i in range(1, len(r)):
        if r[i] > r[b]:
            b = i
        elif r[i] < 0.99 * r[b]:
            return b
    return int(np.argmax(r))


def end_of_first_drop(r, p):
    """Index of the running minimum after the first peak, until the load rises 5 % above it (reloading)."""
    m = p
    for i in range(p + 1, len(r)):
        if r[i] < r[m]:
            m = i
        elif r[i] > 1.05 * r[m]:
            break
    return m


def key_of(label):
    return label.replace(".", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    index = []
    for h5 in sorted(glob.glob(os.path.join(args.campaign_root, "**", "results_*_M100000.h5"), recursive=True)):
        m = RE.search(os.path.basename(h5))
        if not m:
            continue
        beta = float(m["beta"].replace("_", "."))
        rho = float(m["rho"].replace("_", "."))
        case = m["case"]
        eps = float(m["eps"].replace("_", ".")) if m["eps"] else float(EPS_DIR.findall(h5)[-1].replace("_", "."))
        graphs = glob.glob(os.path.join(os.path.dirname(h5), "result_graphs_*_M100000.txt"))
        d = np.loadtxt(graphs[0], comments="#")
        t, R = d[:, 0], np.abs(d[:, 14] + d[:, 15])
        p = first_peak(R)
        q = end_of_first_drop(R, p)
        f = h5py.File(h5, "r")
        keys = list(f["Function"]["s"].keys())
        times = np.array([float(k.replace("_", ".")) for k in keys])
        order = np.argsort(times)
        keys = [keys[i] for i in order]
        times = times[order]
        R_at = np.interp(times, t, R)
        i_ref = int(np.argmin(np.abs(times - 0.003)))
        # stored state after the first crack event: lowest load among the states stored during the first drop;
        # if the drop itself was not stored completely (load still > R_q + 0.1 R1), the first state stored after it
        rr = R_at / R[p]
        during = [i for i in range(len(times)) if t[p] - 1e-9 <= times[i] <= t[q] + 1e-5]
        i_drop = None
        if during:
            i_drop = min(during, key=lambda k: rr[k])
            if rr[i_drop] > R[q] / R[p] + 0.1:
                i_drop = None
        if i_drop is None:
            later = [i for i in range(len(times)) if times[i] > t[q] + 1e-5]
            i_drop = later[0] if later else len(times) - 1
        i_end = len(times) - 1
        want = {"ref": i_ref, "drop": i_drop, "end": i_end}
        all_states = (case == "vary" and abs(beta - 0.01) < 1e-9 and abs(eps - 0.03) < 1e-9)
        out = {"geometry": f["Mesh"]["mesh"]["geometry"][...][:, :2].astype(np.float32),
               "topology": f["Mesh"]["mesh"]["topology"][...].astype(np.int32),
               "E": f["Function"]["E"][keys[i_ref]][...].ravel().astype(np.float32),
               "gc": f["Function"]["gc"][keys[i_ref]][...].ravel().astype(np.float32),
               "sig_ref": f["Function"]["sig"][keys[i_ref]][...].astype(np.float32),
               "times": times, "R_at_times": R_at, "t_peak": t[p], "R_peak": R[p], "t_drop_end": t[q], "R_drop_end": R[q]}
        idx = sorted(set(want.values()) | (set(range(len(times))) if all_states else set()))
        out["state_indices"] = np.array(idx)
        for name, i in want.items():
            out[f"i_{name}"] = i
        for i in idx:
            out[f"s_{i}"] = f["Function"]["s"][keys[i]][...].ravel().astype(np.float32)
            out[f"u_{i}"] = f["Function"]["u"][keys[i]][...][:, :2].astype(np.float32)
        name = f"snap_beta{beta:g}_eps{eps:g}_rho{rho:g}_{case}.npz"
        np.savez_compressed(os.path.join(args.out, name), **out)
        index.append({"file": name, "beta_phi": beta, "beta_s": eps, "rho": rho, "case": case, "t_peak": float(t[p]),
                      "R_peak": float(R[p]), "t_drop_end": float(t[q]), "R_drop_end": float(R[q]),
                      "t_drop_state": float(times[i_drop]), "R_drop_state_over_R1": float(R_at[i_drop] / R[p]),
                      "t_end": float(times[i_end]), "n_states": len(times), "sig_ref_shape": list(out["sig_ref"].shape),
                      "n_nodes": int(out["geometry"].shape[0]), "n_cells": int(out["topology"].shape[0])})
        print(name, f"drop state t={times[i_drop]:.6f} R/R1={R_at[i_drop]/R[p]:.2f}", flush=True)
        f.close()
    with open(os.path.join(args.out, "index.json"), "w") as h:
        json.dump(index, h, indent=1)


if __name__ == "__main__":
    main()
