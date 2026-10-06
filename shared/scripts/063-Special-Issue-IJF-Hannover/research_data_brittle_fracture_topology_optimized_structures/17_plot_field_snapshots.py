#!/usr/bin/env python3
"""Field figures for the revised manuscript from the snapshots written by 16_extract_field_snapshots.py.

Figures (plots/energy_consistent/manuscript/, png + pdf):
  phasefield_s_first_event_eps<beta_s>     phase field after the first crack event (rows E_min/E_max/E_var, cols rho)
  phasefield_s_first_event_beta_phi_eps0_03 same for the E_var structures and the three beta_phi
  crack_evolution_Evar_eps0_03             E_var, beta_phi = 0.01, beta_s = 0.03: after the first, after the second crack
                                           event and at the end displacement (rows rho)
  principal_stress_sig1_eps0_015           first principal stress and its direction at t = 0.003 s (elastic state)
Tables: field_states_M1e5.csv (chosen states), mesh_statistics.csv / table_mesh_statistics.tex
"""
import argparse, csv, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors
import numpy as np

SCALE = 10.0
CASE_LABEL = {"min": r"$\mathbf{E}_{\mathrm{min}}$", "max": r"$\mathbf{E}_{\mathrm{max}}$", "vary": r"$\mathbf{E}_{\mathrm{var}}$"}
FS = 26


def load(folder, beta, eps, rho, case):
    p = folder / f"snap_beta{beta:g}_eps{eps:g}_rho{rho:g}_{case}.npz"
    return np.load(p) if p.exists() else None


def first_event_state(z):
    """Stored state after the first crack event: lowest load among the states stored during the first drop;
    if the drop was not stored, the first state stored after it."""
    T, R = z["times"], z["R_at_times"] / float(z["R_peak"])
    tp, tq, rq = float(z["t_peak"]), float(z["t_drop_end"]), float(z["R_drop_end"]) / float(z["R_peak"])
    during = [i for i in range(len(T)) if tp - 1e-9 <= T[i] <= tq + 1e-5]
    if during:
        i = min(during, key=lambda k: R[k])
        if R[i] <= rq + 0.1:
            return i
    after = [i for i in range(len(T)) if T[i] > tq + 1e-5]
    return after[0] if after else len(T) - 1


def second_event_state(z, i1):
    """End of the largest consecutive load decrease between the first-event state and the end."""
    T, R = z["times"], z["R_at_times"]
    best, best_drop, i = None, 0.0, i1 + 1
    while i < len(T) - 1:
        if R[i] < R[i - 1]:
            j = i
            while j + 1 < len(T) and R[j + 1] < R[j]:
                j += 1
            drop = R[i - 1] - R[j]
            if drop > best_drop:
                best, best_drop = j, drop
            i = j + 1
        else:
            i += 1
    return best if best is not None else len(T) - 1


def triang(z, i):
    xy = z["geometry"].astype(float) + SCALE * z[f"u_{i}"].astype(float)
    return mtri.Triangulation(xy[:, 0], xy[:, 1], z["topology"])


def draw_s(ax, z, i, title):
    tr = triang(z, i)
    im = ax.tripcolor(tr, z[f"s_{i}"], cmap="coolwarm", vmin=0.0, vmax=1.0, shading="gouraud", rasterized=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=FS)
    return im


def ensure_state(z, i):
    return f"s_{i}" in z.files


def s_overview(folder, beta, eps, out, rows_spec, stem, row_labels, records):
    fig, axes = plt.subplots(len(rows_spec), 2, figsize=(22, 2.75 * len(rows_spec) + 1.0))
    im = None
    for r, row in enumerate(rows_spec):
        for c, rho in enumerate((0.3, 0.6)):
            ax = axes[r, c]
            b, case = row
            z = load(folder, b, eps, rho, case)
            if z is None:
                ax.axis("off")
                continue
            i = int(z["i_drop"])
            ratio = float(z["R_at_times"][i] / z["R_peak"])
            im = draw_s(ax, z, i, rf"$\rho={rho:g}$, $u_y={z['times'][i]:.4f}\,$mm")
            records.append({"figure": stem, "beta_phi": b, "beta_s": eps, "rho": rho, "case": case,
                            "u_first_peak": float(z["t_peak"]), "u_state": float(z["times"][i]), "R_state_over_R1": ratio})
        axes[r, 0].text(-0.08, 0.5, row_labels[r], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right",
                        fontsize=FS + 2)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS - 2)
    cb.set_label(r"$s$", fontsize=FS + 2)
    fig.subplots_adjust(left=0.06, right=0.9, wspace=0.05, hspace=0.35)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[INFO]", stem)


def crack_evolution(folder, out, records):
    fig, axes = plt.subplots(2, 3, figsize=(30, 6.8))
    im = None
    for r, rho in enumerate((0.3, 0.6)):
        z = load(folder, 0.01, 0.03, rho, "vary")
        i1 = first_event_state(z)
        i2 = second_event_state(z, i1)
        i3 = len(z["times"]) - 1
        for c, (i, lab) in enumerate(((i1, "after first crack event"), (i2, "after second crack event"), (i3, "end"))):
            ratio = float(z["R_at_times"][i] / z["R_peak"])
            im = draw_s(axes[r, c], z, i, rf"$\rho={rho:g}$, {lab}, $u_y={z['times'][i]:.4f}\,$mm")
            records.append({"figure": "crack_evolution_Evar_eps0_03", "beta_phi": 0.01, "beta_s": 0.03, "rho": rho, "case": "vary",
                            "u_first_peak": float(z["t_peak"]), "u_state": float(z["times"][i]), "R_state_over_R1": ratio})
    cax = fig.add_axes([0.92, 0.15, 0.008, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS - 2)
    cb.set_label(r"$s$", fontsize=FS + 2)
    fig.subplots_adjust(left=0.02, right=0.9, wspace=0.05, hspace=0.3)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"crack_evolution_Evar_eps0_03.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] crack_evolution_Evar_eps0_03")


def principal(sig):
    sxx, sxy, syy = sig[:, 0], 0.5 * (sig[:, 1] + sig[:, 3]), sig[:, 4]
    c = 0.5 * (sxx + syy)
    r = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy ** 2)
    s1 = c + r
    theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    return s1, theta


def principal_stress(folder, out, eps=0.015):
    specs = [("min", 0.3, 0), ("max", 0.3, 1), ("max", 0.6, 1), ("vary", 0.3, 2), ("vary", 0.6, 2)]
    fig, axes = plt.subplots(3, 2, figsize=(22, 9.5))
    for ax in axes.flat:
        ax.axis("off")
    vals = []
    data = {}
    for case, rho, r in specs:
        z = load(folder, 0.01, eps, rho, case)
        i = int(z["i_ref"])
        s1, th = principal(z["sig_ref"].astype(float))
        data[(case, rho)] = (z, i, s1, th)
        vals.append(np.percentile(np.abs(s1), 99))
    vmax = float(np.max(vals))
    im = None
    for case, rho, r in specs:
        z, i, s1, th = data[(case, rho)]
        ax = axes[r, 0 if rho == 0.3 else 1]
        tr = triang(z, i)
        im = ax.tripcolor(tr, facecolors=s1, cmap="RdBu_r", norm=matplotlib.colors.TwoSlopeNorm(vmin=-0.25 * vmax, vcenter=0.0, vmax=vmax), rasterized=True)
        # direction of sigma_1 on a coarse grid, only where sigma_1 is tensile
        tri = z["topology"]
        xy = z["geometry"].astype(float) + SCALE * z[f"u_{i}"].astype(float)
        cent = xy[tri].mean(axis=1)
        gx, gy = np.meshgrid(np.linspace(0.05, 5.95, 60), np.linspace(0.05, 0.95, 10))
        from scipy.spatial import cKDTree
        tree = cKDTree(cent)
        d, k = tree.query(np.c_[gx.ravel(), gy.ravel()])
        k = k[(d < 0.02) & (s1[k] > 0.15 * vmax)]
        L = 0.045
        for kk in np.unique(k):
            cx, cy = cent[kk]
            dx, dy = 0.5 * L * np.cos(th[kk]), 0.5 * L * np.sin(th[kk])
            ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="k", linewidth=1.1)
        ax.set_aspect("equal")
        ax.set_title(rf"$\rho={rho:g}$", fontsize=FS)
    for r, lab in enumerate(("min", "max", "vary")):
        axes[r, 0].text(-0.08, 0.5, CASE_LABEL[lab], transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=FS + 2)
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=FS - 2)
    cb.set_label(r"$\sigma_1$ in N/mm$^2$", fontsize=FS)
    fig.subplots_adjust(left=0.06, right=0.9, wspace=0.05, hspace=0.3)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"principal_stress_sig1_eps{str(eps).replace('.', '_')}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] principal stress, vmax =", vmax)


def mesh_statistics(folder, out):
    rows = []
    seen = set()
    for p in sorted(folder.glob("snap_*_eps0.03_*.npz")):
        z = np.load(p)
        name = p.stem.replace("snap_", "").replace("_eps0.03", "")
        if name in seen:
            continue
        seen.add(name)
        xy, tri = z["geometry"].astype(float), z["topology"]
        edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
        edges = np.unique(edges, axis=0)
        L = np.linalg.norm(xy[edges[:, 0]] - xy[edges[:, 1]], axis=1)
        a = xy[tri]
        area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) - (a[:, 2, 0] - a[:, 0, 0]) * (a[:, 1, 1] - a[:, 0, 1]))
        rows.append({"structure": name, "nodes": len(xy), "triangles": len(tri), "area_mm2": area.sum(),
                     "edge_min": L.min(), "edge_mean": L.mean(), "edge_max": L.max(),
                     "beta_s_min_over_h_mean": 0.015 / L.mean(), "beta_s_max_over_h_mean": 0.06 / L.mean()})
    with open(out / "mesh_statistics.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.5g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("[INFO] mesh statistics:")
    for r in rows:
        print("   ", r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", type=Path, default=Path("fields"))
    ap.add_argument("--out", type=Path, default=Path("plots/energy_consistent/manuscript"))
    args = ap.parse_args()
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    labels = [CASE_LABEL["min"], CASE_LABEL["max"], CASE_LABEL["vary"]]
    for eps in (0.015, 0.03, 0.045, 0.06):
        s_overview(args.fields, 0.01, eps, args.out, [(0.01, "min"), (0.01, "max"), (0.01, "vary")],
                   f"phasefield_s_first_event_eps{str(eps).replace('.', '_')}", labels, records)
    s_overview(args.fields, None, 0.03, args.out, [(0.001, "vary"), (0.01, "vary"), (0.05, "vary")],
               "phasefield_s_first_event_beta_phi_eps0_03",
               [rf"$\beta_\phi={b:g}\,\mathrm{{mm^2}}$" for b in (0.001, 0.01, 0.05)], records)
    crack_evolution(args.fields, args.out, records)
    principal_stress(args.fields, args.out)
    mesh_statistics(args.fields, args.out)
    with open(args.out / "field_states_M1e5.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


if __name__ == "__main__":
    main()
