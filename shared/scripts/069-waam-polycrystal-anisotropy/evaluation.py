#!/usr/bin/env python3
"""
Evaluate and plot the WAAM polycrystal FE results.

Reads the outputs written by the FE runs in this folder:
  * Chom_<MAT>.json        (KUBC homogenization -> 6x6 effective stiffness)
  * Emodul_<MAT>_<O>.json  (uniaxial tensile test -> apparent E, O in V/H/45deg)

and produces (into ./report/ by default):
  * fig1_uniaxial_E.png      grouped bar chart of apparent E (V/H/45deg)
  * fig2_crosscheck.png      uniaxial bar E vs. RVE directional E (validation)
  * fig3_Chom_heatmap.png    effective stiffness matrices as heatmaps
  * summary.csv / summary.md tables of all derived quantities

Usage:  python3 evaluation.py [--dir .] [--out report] [--materials 316L 17-4PH]
No dolfinx needed - pure post-processing (numpy + matplotlib).
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORI = ["V", "H", "45deg"]
ORI_LABEL = {"V": "V (load ∥ build)", "H": "H (load ⊥ build)", "45deg": "45°"}
COLOR = {"316L": "#2c6fbb", "17-4PH": "#c1121f"}
VLABEL = ["xx", "yy", "zz", "yz", "xz", "xy"]


# ---------------------------------------------------------------------------
def discover_materials(d):
    mats = sorted({os.path.basename(p)[5:-5] for p in glob.glob(os.path.join(d, "Chom_*.json"))})
    return mats


def directional_E(Csym):
    S = np.linalg.inv(Csym)
    return [1.0 / S[i, i] for i in range(3)], S


def iso_equiv(C):
    lam = np.mean([C[0, 1], C[0, 2], C[1, 0], C[2, 0], C[1, 2], C[2, 1]])
    mu = np.mean([C[3, 3], C[4, 4], C[5, 5]])
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    nu = lam / (2 * (lam + mu))
    return E, nu, lam, mu


def coupling_metric(C):
    """Magnitude of the off-diagonal blocks that vanish for an orthotropic C."""
    idx = [(i, j) for i in range(6) for j in range(6)
           if (i < 3 <= j) or (j < 3 <= i) or (i >= 3 and j >= 3 and i != j)]
    vals = np.array([abs(C[i, j]) for i, j in idx])
    return vals.mean(), vals.max()


def load_all(d, materials):
    chom, uni = {}, {}
    for m in materials:
        p = os.path.join(d, f"Chom_{m}.json")
        if os.path.isfile(p):
            chom[m] = json.load(open(p))
        uni[m] = {}
        for o in ORI:
            pe = os.path.join(d, f"Emodul_{m}_{o}.json")
            if os.path.isfile(pe):
                uni[m][o] = json.load(open(pe))
    return chom, uni


# ---------------------------------------------------------------------------
def fig_uniaxial(uni, materials, out):
    mats = [m for m in materials if uni.get(m)]
    if not mats:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(ORI))
    w = 0.8 / max(len(mats), 1)
    for i, m in enumerate(mats):
        vals = [uni[m].get(o, {}).get("E_apparent_GPa", np.nan) for o in ORI]
        b = ax.bar(x + (i - (len(mats) - 1) / 2) * w, vals, w, label=m,
                   color=COLOR.get(m, None), edgecolor="k", linewidth=0.6)
        ax.bar_label(b, fmt="%.0f", fontsize=9, padding=2)
    ax.set_xticks(x); ax.set_xticklabels([ORI_LABEL[o] for o in ORI])
    ax.set_ylabel("apparent Young's modulus E [GPa]")
    ax.set_title("Numerical uniaxial tensile test — directional E-modulus")
    ax.set_ylim(0, 250); ax.legend(title="steel"); ax.grid(axis="y", ls=":", alpha=.5)
    fig.tight_layout()
    p = os.path.join(out, "fig1_uniaxial_E.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_crosscheck(uni, chom, materials, out):
    mats = [m for m in materials if uni.get(m) and m in chom]
    if not mats:
        return None
    fig, ax = plt.subplots(1, len(mats), figsize=(5 * len(mats), 4.2), sharey=True, squeeze=False)
    for k, m in enumerate(mats):
        a = ax[0][k]
        vals = [uni[m].get(o, {}).get("E_apparent_GPa", np.nan) for o in ORI]
        a.bar(["V", "H", "45°"], vals, width=0.5, color=COLOR.get(m), alpha=.85,
              edgecolor="k", label="uniaxial bar")
        Edir, _ = directional_E(np.array(chom[m]["Chom_sym"]))
        ez, ext = Edir[2], np.mean(Edir[:2])
        a.hlines([ez], -0.4, 0.4, color="k", ls="--", lw=2)
        a.hlines([ext], 0.6, 1.4, color="k", ls="--", lw=2, label="RVE homogenization")
        a.annotate(f"E_z={ez:.0f}", (0, ez), ha="center", va="bottom", fontsize=8)
        a.annotate(f"E_⊥={ext:.0f}", (1, ext), ha="center", va="bottom", fontsize=8)
        a.set_title(m); a.grid(axis="y", ls=":", alpha=.5)
    ax[0][0].set_ylabel("E [GPa]"); ax[0][0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Cross-check: uniaxial bar vs. RVE homogenization (dashed)")
    fig.tight_layout()
    p = os.path.join(out, "fig2_crosscheck.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_heatmap(chom, materials, out):
    mats = [m for m in materials if m in chom]
    if not mats:
        return None
    fig, ax = plt.subplots(1, len(mats), figsize=(5.5 * len(mats), 4.6), squeeze=False)
    for k, m in enumerate(mats):
        a = ax[0][k]
        C = np.array(chom[m]["Chom_sym"])
        vmax = np.abs(C).max()
        im = a.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for i in range(6):
            for j in range(6):
                a.text(j, i, f"{C[i, j]:.0f}", ha="center", va="center", fontsize=7,
                       color="k" if abs(C[i, j]) < 0.6 * vmax else "w")
        a.set_xticks(range(6)); a.set_xticklabels(VLABEL)
        a.set_yticks(range(6)); a.set_yticklabels(VLABEL)
        a.set_title(f"{m}: effective stiffness Chom [GPa]")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.tight_layout()
    p = os.path.join(out, "fig3_Chom_heatmap.png"); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def write_tables(uni, chom, materials, out):
    rows = []
    for m in materials:
        if m in chom:
            C = np.array(chom[m]["Chom_sym"])
            Edir, _ = directional_E(C)
            Eiso, nuiso, lam, mu = iso_equiv(C)
            cmean, cmax = coupling_metric(C)
            spread = 100 * (max(Edir) - min(Edir)) / np.mean(Edir)
            rows.append(dict(material=m, method="RVE homogenization (KUBC)",
                             E_x=Edir[0], E_y=Edir[1], E_z=Edir[2], E_iso=Eiso,
                             nu_iso=nuiso, E_spread_pct=spread,
                             coupling_mean=cmean, coupling_max=cmax))
        for o in ORI:
            if uni.get(m, {}).get(o):
                d = uni[m][o]
                rows.append(dict(material=m, method=f"uniaxial {o}",
                                 E_x=d["E_apparent_GPa"], E_y=np.nan, E_z=np.nan,
                                 E_iso=np.nan, nu_iso=d.get("nu_xy"),
                                 E_spread_pct=np.nan, coupling_mean=np.nan, coupling_max=np.nan))
    # CSV
    keys = ["material", "method", "E_x", "E_y", "E_z", "E_iso", "nu_iso",
            "E_spread_pct", "coupling_mean", "coupling_max"]
    with open(os.path.join(out, "summary.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if (isinstance(r.get(k), float) and np.isnan(r.get(k)))
                             else (f"{r.get(k):.3f}" if isinstance(r.get(k), float) else str(r.get(k, "")))
                             for k in keys) + "\n")
    # Markdown
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("| material | method | E_x | E_y | E_z | E_iso | nu | spread% | coupl.max |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            def g(k, fmt="{:.1f}"):
                v = r.get(k)
                return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else fmt.format(v)
            f.write(f"| {r['material']} | {r['method']} | {g('E_x')} | {g('E_y')} | {g('E_z')} | "
                    f"{g('E_iso')} | {g('nu_iso','{:.3f}')} | {g('E_spread_pct')} | {g('coupling_max')} |\n")
    return rows


def fig_waam_geometry(out):
    """Illustrative schematic (no data): WAAM directions + the transverse-isotropy
    assumption vs. the in-plane super-elongation seen in the horizontal section.
    Two views of the section PERPENDICULAR to the build direction z."""
    from matplotlib.patches import Rectangle, FancyArrowPatch
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    panels = [("Fruehere Annahme: transversal isotrop\n(in der Wandebene gleichachsig)", 6, 6),
              ("Modell jetzt = Horizontalschliff:\northotrop, entlang Schweissrichtung gestreckt", 3, 9)]
    for k, (title, ncol, nrow) in enumerate(panels):
        a = ax[k]
        w, h = 1.0 / ncol, 1.0 / nrow
        for i in range(ncol):
            for j in range(nrow):
                jx, jy = 0.12 * w * rng.standard_normal(), 0.12 * h * rng.standard_normal()
                a.add_patch(Rectangle((i * w + 0.02 * w + jx, j * h + 0.02 * h + jy),
                                      w * 0.96, h * 0.96, facecolor=plt.cm.Pastel1((i + 2 * j) % 9),
                                      edgecolor="k", lw=0.8))
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=10)
        a.set_xlabel("Schweißrichtung  x  →")
        a.set_ylabel("Wandnormale  y  →")
    fig.suptitle("Schnitt senkrecht zur Aufbaurichtung (z zeigt aus der Bildebene heraus).  "
                 "Entlang z sind die Koerner kolumnar (lang gestreckt).", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "fig0_waam_geometry.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_microstructure(neper_dir, materials, out):
    """Compose the Neper tessellation renders (colored by orientation) of the
    generated RVEs side by side, whitespace cropped."""
    def crop(path):
        img = plt.imread(path)
        g = img[..., :3].mean(axis=2)
        m = g < 0.97
        r = np.where(m.any(axis=1))[0]; c = np.where(m.any(axis=0))[0]
        if len(r) == 0 or len(c) == 0:
            return img
        pad = 6
        return img[max(r.min() - pad, 0):r.max() + pad, max(c.min() - pad, 0):c.max() + pad]
    items = []
    label = {"316L": "316L RVE — kolumnar\n(Körner entlang der Aufbaurichtung gestreckt)",
             "17-4PH": "17-4PH RVE — equiaxed\n(gleichachsige Körner)"}
    for m in materials:
        pj = os.path.join(neper_dir, f"params_{m}.json")
        if not os.path.isfile(pj):
            continue
        n = json.load(open(pj))["n_grains"]
        p = os.path.join(neper_dir, f"waam_{m}_n{n}_tess.png")
        if os.path.isfile(p):
            items.append((m, crop(p)))
    if not items:
        return None
    fig, ax = plt.subplots(1, len(items), figsize=(5.2 * len(items), 5.2), squeeze=False)
    for k, (m, img) in enumerate(items):
        a = ax[0][k]
        a.imshow(img); a.set_axis_off()
        a.set_title(label.get(m, m), fontsize=10)
    fig.suptitle("Generierte 3D-Mikrostruktur (Neper-Tessellation, Färbung nach Kornorientierung)",
                 fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "fig_microstructure.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    return p


def fig_rve_frame(out):
    """Sketch of the RVE coordinate frame: x = weld, y = wall-normal, z = build,
    with one columnar grain (long in z, equiaxed in the x-y plane)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    # RVE box edges (unit cube)
    c = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    for i, a in enumerate(c):
        for b in c[i + 1:]:
            if np.isclose(np.linalg.norm(a - b), 1.0):
                ax.plot(*zip(a, b), color="0.7", lw=1)
    # one columnar grain (prolate spheroid): rz long (build), rx = ry (equiaxed
    # in the weld / wall-normal plane), aspect ~3.7 as measured for 316L
    u = np.linspace(0, 2 * np.pi, 40); v = np.linspace(0, np.pi, 20)
    rx, ry, rz = 0.11, 0.11, 0.40
    gx = 0.5 + rx * np.outer(np.cos(u), np.sin(v))
    gy = 0.5 + ry * np.outer(np.sin(u), np.sin(v))
    gz = 0.5 + rz * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(gx, gy, gz, color="#2c6fbb", alpha=0.55, linewidth=0)
    # axis arrows from origin
    L = 1.28
    for vec, col, lab in [((L, 0, 0), "#c1121f", "x  Schweißrichtung"),
                          ((0, L, 0), "#1a7f37", "y  Wandnormale"),
                          ((0, 0, L), "#000000", "z  Aufbaurichtung")]:
        ax.quiver(0, 0, 0, *vec, color=col, lw=2.5, arrow_length_ratio=0.08)
        ax.text(vec[0] * 1.02, vec[1] * 1.02, vec[2] * 1.02, lab, color=col, fontsize=9)
    ax.text(0.5, 0.5, 0.95, "kolumnares Korn\n(k ≈ 3.7 entlang z)", ha="center", fontsize=8, color="#2c6fbb")
    ax.set_xlim(0, 1.3); ax.set_ylim(0, 1.3); ax.set_zlim(0, 1.3)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(elev=16, azim=-60)
    ax.set_title("RVE-Rahmen: Aufbaurichtung → z-Achse\n(Textur wird dorthin rotiert)", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "fig_rve_frame.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_phases(out):
    """Illustrative sketch of the two crystal structures (unit cells):
    FCC (austenite, 316L) vs. BCC (martensite, 17-4PH)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = np.array([[.5, .5, 0], [.5, .5, 1], [.5, 0, .5], [.5, 1, .5], [0, .5, .5], [1, .5, .5]])
    body = np.array([[.5, .5, .5]])
    edges = [(a, b) for i, a in enumerate(corners) for b in corners[i + 1:]
             if np.isclose(np.linalg.norm(a - b), 1.0)]
    fig = plt.figure(figsize=(9.5, 4.6))
    for k, (title, extra, ecol) in enumerate([
            ("FCC (kfz) — Austenit (316L)\nEckatome + Flächenmitten", faces, "#c1121f"),
            ("BCC (krz) — Martensit (17-4PH)\nEckatome + Raummitte", body, "#2c6fbb")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for a, b in edges:
            ax.plot(*zip(a, b), color="0.6", lw=1)
        ax.scatter(*corners.T, s=260, color="0.35", depthshade=True, label="Eckatome")
        ax.scatter(*extra.T, s=260, color=ecol, depthshade=True, label="zusätzliche Atome")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=32)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)
    fig.suptitle("Kristallstrukturen der Phasen (Elementarzelle). Die Anordnung der "
                 "Atome bestimmt die anisotrope Einkristall-Steifigkeit.", fontsize=10)
    fig.tight_layout()
    p = os.path.join(out, "fig_phases.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    return p


def microstructure_matching(neper_dir, materials, out):
    """Statistical matching of the GENERATED tessellation vs. the EBSD-fitted
    target grain-size distribution. Reads params_<MAT>.json + waam_<MAT>_n<N>.stcell
    from the Neper pipeline folder; writes fig4_size_matching.png +
    summary_microstructure.md and returns the matching rows."""
    mats = [m for m in materials
            if os.path.isfile(os.path.join(neper_dir, f"params_{m}.json"))]
    if not mats:
        return []
    fig, ax = plt.subplots(1, len(mats), figsize=(5.4 * len(mats), 4.2), squeeze=False)
    rows = []
    for k, m in enumerate(mats):
        p = json.load(open(os.path.join(neper_dir, f"params_{m}.json")))
        n = p["n_grains"]
        kel = p["neper"].get("scale_product", p["elongation"]["k_used"])
        med_t = p["transverse"]["d3D_median_um"]
        mean_t = p["transverse"]["d3D_mean_um"]
        cv_t = p["neper"].get("morpho_cv", p["transverse"]["width2D_cv"])
        stc = os.path.join(neper_dir, f"waam_{m}_n{n}.stcell")
        if not os.path.isfile(stc):
            continue
        vol = np.loadtxt(stc, usecols=1)               # cell volume (final, stretched)
        dt = (6.0 * vol / (np.pi * kel)) ** (1 / 3)    # transverse-equiv diameter
        med_a, mean_a, cv_a = float(np.median(dt)), float(dt.mean()), float(dt.std() / dt.mean())
        a = ax[0][k]
        bins = np.geomspace(dt.min() * 0.7, dt.max() * 1.3, 22)
        a.hist(dt, bins=bins, density=True, alpha=.6, color=COLOR.get(m),
               edgecolor="k", label="generiert (RVE)")
        s_ln, mu = np.sqrt(np.log(1 + cv_t ** 2)), np.log(med_t)
        xx = np.geomspace(bins[0], bins[-1], 250)
        pdf = np.exp(-(np.log(xx) - mu) ** 2 / (2 * s_ln ** 2)) / (xx * s_ln * np.sqrt(2 * np.pi))
        a.plot(xx, pdf, "k-", lw=2, label="Ziel-Lognormal (EBSD-Fit)")
        a.axvline(med_a, color=COLOR.get(m), ls="--")
        a.axvline(med_t, color="k", ls=":")
        a.set_xscale("log")
        a.set_xlabel("transversaler Korndurchmesser d_t [µm]")
        a.set_title(f"{m}: Median {med_a:.0f} vs {med_t:.0f} µm | CV {cv_a:.2f} vs {cv_t:.2f}")
        a.legend(fontsize=8)
        rows.append(dict(material=m, med_target=med_t, med_achieved=med_a,
                         mean_target=mean_t, mean_achieved=mean_a,
                         cv_target=cv_t, cv_achieved=cv_a))
    ax[0][0].set_ylabel("Dichte")
    fig.suptitle("Statistisches Matching: generierte Struktur vs. EBSD-Ziel (Korngröße)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_size_matching.png"), dpi=150)
    plt.close(fig)
    with open(os.path.join(out, "summary_microstructure.md"), "w") as f:
        f.write("| Stahl | Median Ziel | Median erreicht | Mittel Ziel | Mittel erreicht | CV Ziel | CV erreicht |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['material']} | {r['med_target']:.0f} | {r['med_achieved']:.0f} | "
                    f"{r['mean_target']:.0f} | {r['mean_achieved']:.0f} | "
                    f"{r['cv_target']:.2f} | {r['cv_achieved']:.2f} |\n")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--dir", default=here, help="folder with Chom_*/Emodul_* json")
    ap.add_argument("--out", default=os.path.join(here, "report"), help="output folder")
    ap.add_argument("--materials", nargs="*", default=None)
    ap.add_argument("--neper-dir", default=None,
                    help="Neper pipeline folder (adds microstructure matching + EBSD-fit plots)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    materials = args.materials or discover_materials(args.dir)
    if not materials:
        raise SystemExit(f"no Chom_*.json found in {args.dir}")
    chom, uni = load_all(args.dir, materials)
    print("materials:", materials)

    for fn in (fig_waam_geometry(args.out),
               fig_rve_frame(args.out),
               fig_phases(args.out),
               fig_uniaxial(uni, materials, args.out),
               fig_crosscheck(uni, chom, materials, args.out),
               fig_heatmap(chom, materials, args.out)):
        if fn:
            print("wrote", os.path.relpath(fn, here))
    write_tables(uni, chom, materials, args.out)
    print("wrote summary.csv, summary.md")

    if args.neper_dir:
        import shutil
        mfn = fig_microstructure(args.neper_dir, materials, args.out)
        if mfn:
            print("wrote", os.path.relpath(mfn, here))
        rows = microstructure_matching(args.neper_dir, materials, args.out)
        if rows:
            print("wrote fig4_size_matching.png, summary_microstructure.md")
            for r in rows:
                print(f"  {r['material']:7s} size match: median {r['med_achieved']:.0f}/"
                      f"{r['med_target']:.0f} um, CV {r['cv_achieved']:.2f}/{r['cv_target']:.2f}")
        for m in materials:                            # copy EBSD-vs-fit diagnostics
            src = os.path.join(args.neper_dir, f"fit_ebsd_targets_{m}.png")
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(args.out, f"ebsd_fit_{m}.png"))
                print(f"  copied ebsd_fit_{m}.png")
    # console recap
    for m in materials:
        if m in chom:
            Edir, _ = directional_E(np.array(chom[m]["Chom_sym"]))
            print(f"  {m} RVE  E_x,E_y,E_z = {Edir[0]:.1f},{Edir[1]:.1f},{Edir[2]:.1f} GPa")
        for o in ORI:
            if uni.get(m, {}).get(o):
                print(f"  {m} uniaxial {o:5s} E = {uni[m][o]['E_apparent_GPa']:.1f} GPa")


if __name__ == "__main__":
    main()
