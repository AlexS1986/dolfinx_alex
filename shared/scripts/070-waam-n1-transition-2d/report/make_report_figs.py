#!/usr/bin/env python3
"""
Figures for the 070 report that depend ONLY on input data and the material
assignment - no FE solve. (The FE result figure is produced separately by
../make_figures.py from the dolfinx output.)

  fig_roi_map.png        annotated EBSD map with ROI + transition box
  fig_reconstruction.png original IPF-Z vs. reconstructed grain map (ROI)
  fig_grain_Ex.png       per-grain E_x map + histogram per zone
  fig_bounds.png         Voigt/Reuss bracket per zone (+ FE result if present)
  fig_fe_fields.png      sigma_xx and eps_xx maps from the dolfinx run
  fig_E_profile.png      local E(x) profile + numerator/denominator decomposition
  fig_dic_profile.png    experimental full-gauge DIC profile
  fig_grain_Ex_variants  per-grain E_x map for s=1 and both scaled variants
  fig_averaging.png      strip averages of the per-grain moduli vs. the FE result
  fig_mesh_zoom.png      zoom: the regular Q1 mesh on top of the grain structure
  fig_strips.png         sketch of the strip subdivision used for the E(x) profile
  fig_ferrite.png        ferrite content vs. local E on the same specimen
  profile_stats.json     numbers quoted in the report's worked example

The FE-dependent figures read `fields_roi.npz` if present, otherwise the XDMF
companion `ps_roi.h5` of any dolfinx run.

Usage: python3 report/make_report_figs.py [--bmp <annotated bmp>] [--src <raw bmp>]
"""
import argparse, json, os, sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import materials_2d as M2                                    # noqa: E402
from preprocess_ebsd_to_grid import S_UM_PER_PX              # noqa: E402

C_EXP, C_FE, C_V, C_R = '#4c72b0', '#c44e52', '#8172b3', '#dd8452'
ROI = (30., 1339., 2811., 2225.)
X_TRANS_MAP = (666.6, 1494.5)
EXP_ZONE = {'17-4PH': (201.7, 2.0), 'transition': (232.4, 13.0), '316L': (162.7, 2.0)}
ZONES = [(0, '17-4PH'), (1, 'Übergang'), (2, '316L')]

ap = argparse.ArgumentParser()
ap.add_argument('--bmp', default=None, help='..._mit_AR_Bereich.bmp (annotiert)')
ap.add_argument('--src', default=None, help='WAAM_N=1_A12D_Uebergangsbereich.bmp')
args = ap.parse_args()

d = np.load(os.path.join(ROOT, 'micro_roi.npz'))
euler, phase, gid, zone = d['euler_deg'], d['phase'], d['grain_id'], d['zone']
meta = json.loads(str(d['meta']))
step = meta['step_um']
ny, nx = phase.shape
Lx, Ly = nx * step, ny * step
EXT = [0, Lx, Ly, 0]
XT = tuple(x - ROI[0] for x in X_TRANS_MAP)

cfg = M2.load_config(here=ROOT)
C, Ex, s_map, info = M2.build_cell_tensors(
    euler, phase, gid, zone, (np.arange(nx) + 0.5) * step, cfg, verbose=False)



def load_fe_fields(tag='roi', h5=None):
    """Cell fields of a dolfinx run, mapped onto the (ny, nx) microstructure grid.

    Prefers `fields_<tag>.npz` (written by the current solve_plane_stress.py);
    falls back to the XDMF companion file `ps_<tag>.h5`, which every dolfinx run
    produces regardless of script version. Returns None if neither exists.
    """
    p = os.path.join(ROOT, f'fields_{tag}.npz')
    if os.path.isfile(p):
        F = np.load(p)
        return {'sig_xx': F['sig_xx'], 'eps_xx': F['eps_xx'], 'src': 'fields npz'}
    p = h5 or os.path.join(ROOT, f'ps_{tag}.h5')
    if not os.path.isfile(p):
        return None
    import h5py
    with h5py.File(p, 'r') as f:
        geo = f['Mesh/mesh/geometry'][:]
        topo = f['Mesh/mesh/topology'][:]
        mid = geo[topo].mean(axis=1)
        ci = np.clip((mid[:, 0] / step).astype(int), 0, nx - 1)
        cj = np.clip(ny - 1 - (mid[:, 1] / step).astype(int), 0, ny - 1)
        out = {}
        for k in ('sig_xx', 'eps_xx'):
            a = np.zeros((ny, nx))
            a[cj, ci] = f[f'Function/{k}/0'][:, 0]
            out[k] = a
    out['src'] = 'ps h5'
    return out


def bin_profile(sig, eps, bin_um=100.0):
    """E(x_k) = sum(sig_xx)/sum(eps_xx) over ~bin_um wide strips (as in the solver)."""
    nbin = max(int(round(Lx / bin_um)), 10)
    edges = np.linspace(0.0, Lx, nbin + 1)
    xcell = (np.arange(nx) + 0.5) * step
    ib = np.clip(np.digitize(xcell, edges) - 1, 0, nbin - 1)
    xk = 0.5 * (edges[:-1] + edges[1:])
    S = np.array([sig[:, ib == k].sum() for k in range(nbin)])
    P = np.array([eps[:, ib == k].sum() for k in range(nbin)])
    return xk, S / P, S, P, edges, ib



# s(x)-Varianten des Modells: Tag -> (Beschriftung, Funktion, Farbe)
SVAR = [
    ('roi_s133', 's = 1,33 (konstant)', lambda x: 1.33, '#dd8452'),
    ('roi_gauss', r's(x) = 1 + 0,50·exp[−((x−1050)/350)²]',
     lambda x: 1 + 0.50 * np.exp(-((x - 1050.) / 350.) ** 2), '#55a868'),
]


def variant_profile(tag, sfun, xk, E1, zone_k):
    """E(x) einer s-Variante.

    Liegt ein dolfinx-Lauf vor (fields_<tag>.npz oder ps_<tag>.h5), wird dessen
    Profil verwendet ('FE'). Sonst wird es aus der s=1-Loesung abgeleitet ('abg.'):
    In der Serienanordnung ist die mittlere Spannung jedes Querschnitts durch das
    Gleichgewicht festgelegt, sodass das Skalieren der Zonensteifigkeit den
    Streifenmodul im selben Verhaeltnis anhebt, E_k -> s_k E_k. Vernachlaessigt
    wird dabei die Umverteilung der Dehnung INNERHALB eines Streifens.
    """
    fe = load_fe_fields(tag)
    if fe is not None:
        _, Ek, *_ = bin_profile(fe['sig_xx'], fe['eps_xx'])
        return Ek, 'FE'
    sk = np.array([sfun(x) if z == 1 else 1.0 for x, z in zip(xk, zone_k)])
    return sk * E1, 'abg.'

# ---- fig_roi_map -----------------------------------------------------------
if args.bmp and os.path.isfile(args.bmp):
    im = Image.open(args.bmp).convert('RGB')
    im.thumbnail((1600, 1600))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(np.asarray(im))
    ax.axis('off')
    ax.set_title('EBSD-Map des Übergangsbereichs mit Auswerte-ROI (schwarz)\n'
                 'und Übergangszone (grün); Aufbaurichtung = −x', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_roi_map.png'), dpi=150, bbox_inches='tight')
    print('wrote fig_roi_map.png')

# ---- fig_reconstruction ----------------------------------------------------
rec = np.asarray(Image.open(os.path.join(ROOT, 'micro_roi_ipf.png')))
if args.src and os.path.isfile(args.src):
    src = np.asarray(Image.open(args.src).convert('RGB'))
    H, W, _ = src.shape
    xc = np.arange(ROI[0] + step / 2, ROI[2], step)
    yc = np.arange(ROI[1] + step / 2, ROI[3], step)
    px = np.clip((xc / S_UM_PER_PX).astype(int), 0, W - 1)
    py = np.clip((yc / S_UM_PER_PX).astype(int), 0, H - 1)
    orig = src[np.ix_(py, px)]
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.2))
    ZBOX = (250 * step, 95 * step, 42 * step, 30 * step)   # = Ausschnitt fig_mesh_zoom
    for ax, img, t in zip(axs, [orig, rec],
                          ['(a) EBSD-Originalscan (IPF-Z), ROI',
                           '(b) Rekonstruierte Kornkarte (Modelleingang)']):
        ax.imshow(img, extent=EXT, aspect='equal')
        for xb in XT:
            ax.axvline(xb, color='k', lw=1.1, ls='--')
        ax.set_title(t, fontsize=10, loc='left')
        ax.set_ylabel('y [µm]')
    axs[1].add_patch(plt.Rectangle(ZBOX[:2], ZBOX[2], ZBOX[3], fill=False,
                                   ec='k', lw=2.0))
    axs[1].annotate('Netzausschnitt', (ZBOX[0] + ZBOX[2], ZBOX[1] + ZBOX[3] / 2),
                    textcoords='offset points', xytext=(16, 0), va='center',
                    fontsize=8.5, bbox=dict(fc='white', alpha=0.9, ec='0.4', pad=2),
                    arrowprops=dict(arrowstyle='->', lw=1.1))
    axs[1].set_xlabel('x [µm] (Lastachse)')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_reconstruction.png'), dpi=150, bbox_inches='tight')
    print('wrote fig_reconstruction.png')

# ---- fig_grain_Ex ----------------------------------------------------------
fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.03], wspace=0.28)
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(Ex, extent=EXT, aspect='equal', cmap='plasma')
for xb in XT:
    ax.axvline(xb, color='w', lw=1.1, ls='--')
ax.set_xlabel('x [µm] (Lastachse)')
ax.set_ylabel('y [µm]')
ax.set_title(r'(a) $E_x$ jedes Korns aus dessen eigenem, rotiertem Tensor '
             r'— Fall s(x) = 1', fontsize=10, loc='left')
plt.colorbar(im, cax=fig.add_subplot(gs[0, 2]), label=r'$E_x$ [GPa]')
ax = fig.add_subplot(gs[0, 1])
for z, nm in ZONES:
    ax.hist(Ex[zone == z].ravel(), bins=60, range=(80, 310), histtype='step',
            lw=1.6, density=True, label=nm)
ax.set_xlabel(r'$E_x$ des Korns [GPa]')
ax.set_ylabel('Dichte (flächengewichtet)')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_title('(b) Verteilung je Zone, s(x) = 1', fontsize=10, loc='left')
fig.savefig(os.path.join(HERE, 'fig_grain_Ex.png'), dpi=150, bbox_inches='tight')
print('wrote fig_grain_Ex.png')

# ---- fig_bounds ------------------------------------------------------------
fe = None
p = os.path.join(ROOT, 'E_roi.json')
if os.path.isfile(p):
    fe = json.load(open(p))
key = {'17-4PH': 'zone_17-4PH', 'Übergang': 'zone_transition', '316L': 'zone_316L'}
ekey = {'17-4PH': '17-4PH', 'Übergang': 'transition', '316L': '316L'}
fig, ax = plt.subplots(figsize=(7.6, 4.4))
xs = np.arange(3)
for i, (z, nm) in enumerate(ZONES):
    m = zone == z
    Cv = C[m].mean(axis=0)
    Sr = np.linalg.inv(C[m]).mean(axis=0)
    Ev, Er = 1 / np.linalg.inv(Cv)[0, 0], 1 / Sr[0, 0]
    ax.vlines(i, Er, Ev, color=C_V, lw=9, alpha=0.35,
              label='Voigt–Reuss-Schranken' if i == 0 else None)
    ax.plot([i - .18, i + .18], [Ev, Ev], color=C_V, lw=2)
    ax.plot([i - .18, i + .18], [Er, Er], color=C_V, lw=2)
    if fe:
        v = fe[key[nm]]['E_local_GPa']
        ax.plot(i, v, 'o', color=C_FE, ms=9, zorder=5,
                label='FE (dolfinx), s(x)=1' if i == 0 else None)
        ax.annotate(f'{v:.0f}', (i, v), textcoords='offset points',
                    xytext=(0, 11), ha='center', fontsize=8, color=C_FE)
    m_, s_ = EXP_ZONE[ekey[nm]]
    ax.errorbar(i + 0.28, m_, yerr=s_, fmt='s', color=C_EXP, ms=7, capsize=4,
                label='Experiment (DIC)' if i == 0 else None)
    ax.annotate(f'{m_:.0f}', (i + 0.28, m_), textcoords='offset points',
                xytext=(10, -3), fontsize=8, color=C_EXP)
ax.set_xticks(xs)
ax.set_xticklabels([nm for _, nm in ZONES])
ax.set_xlim(-0.45, 2.72)
ax.set_ylabel(r'$E_x$ [GPa]')
ax.set_ylim(140, 262)
ax.grid(axis='y', alpha=0.3)
ax.legend(fontsize=8, loc='upper left')
ax.set_title('Analytische Schranken der Kornstruktur, FE-Ergebnis und Messwert',
             fontsize=10, loc='left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_bounds.png'), dpi=150, bbox_inches='tight')
print('wrote fig_bounds.png')

# ---- FE-abhaengige Abbildungen (nur wenn ein dolfinx-Lauf vorliegt) --------
FE = load_fe_fields('roi')
if FE is None:
    print('kein dolfinx-Ergebnis gefunden (fields_roi.npz / ps_roi.h5) - '
          'FE-Abbildungen uebersprungen')
else:
    print(f'FE-Felder geladen aus: {FE["src"]}')
    sig, eps = FE['sig_xx'], FE['eps_xx']
    xk, Ek, Ssum, Psum, edges, ib = bin_profile(sig, eps)
    KH = int(np.argmin(np.abs(xk - 1241.6)))          # hervorgehobener Beispielstreifen
    xa, xb = edges[KH], edges[KH + 1]

    # --- fig_fe_fields: sigma_xx und eps_xx nebeneinander -------------------
    fig, axs = plt.subplots(2, 1, figsize=(11, 7.4))
    for ax, fld, lab, cm, sc in [
            (axs[0], sig * 1000.0, r'$\sigma_{xx}$ [MPa]', 'viridis', 1),
            (axs[1], eps * 1e3, r'$\varepsilon_{xx}$ [$\perthousand$]', 'cividis', 1)]:
        im = ax.imshow(fld, extent=EXT, aspect='equal', cmap=cm,
                       vmin=np.percentile(fld, 2), vmax=np.percentile(fld, 98))
        for x_ in XT:
            ax.axvline(x_, color='w', lw=1.1, ls='--')
        ax.add_patch(plt.Rectangle((xa, 0), xb - xa, Ly, fill=False,
                                   ec='red', lw=1.6))
        plt.colorbar(im, ax=ax, label=lab, fraction=0.03, pad=0.02)
        ax.set_ylabel('y [µm]')
    axs[0].set_title(r'(a) $\sigma_{xx}$ — korngenaue Streuung, aber in jedem '
                     'Streifen derselbe Mittelwert', fontsize=10, loc='left')
    axs[1].set_title(r'(b) $\varepsilon_{xx}$ — hier steckt die Ortsabhängigkeit '
                     'von E(x)', fontsize=10, loc='left')
    axs[1].set_xlabel('x [µm] (Lastachse)')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_fe_fields.png'), dpi=150, bbox_inches='tight')
    print('wrote fig_fe_fields.png')

    # --- fig_E_profile: Profil + Zerlegung in Zähler/Nenner ---------------
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.8), sharex=True,
                            gridspec_kw={'height_ratios': [1.5, 0.62, 0.95]})
    # Zone je Streifen (Mehrheitsentscheid) fuer die s-Varianten
    zone_k = np.array([np.bincount(zone[:, ib == k].ravel().astype(int),
                                   minlength=3).argmax() for k in range(len(xk))])
    ax = axs[0]
    ax.plot(xk / 1000.0, Ek, 'o-', color=C_FE, lw=1.8, ms=4.5,
            label='s(x) = 1 — reine Mikrostruktur (FE)')
    for tag, lab, sfun, col in SVAR:
        Ev, kind = variant_profile(tag, sfun, xk, Ek, zone_k)
        ax.plot(xk / 1000.0, Ev, 'o-' if kind == 'FE' else 'o--', color=col,
                lw=1.8, ms=4, alpha=0.95,
                label=f'{lab} — {"FE" if kind == "FE" else "aus s=1 abgeleitet"}')
    seg = {'17-4PH': (0.0, XT[0]), 'transition': XT, '316L': (XT[1], Lx)}
    band = line = None
    for z, (a_, b_) in seg.items():
        m_, s_ = EXP_ZONE[z]
        band = ax.fill_between([a_ / 1000.0, b_ / 1000.0], m_ - s_, m_ + s_,
                               color=C_EXP, alpha=0.25, lw=0)
        line, = ax.plot([a_ / 1000.0, b_ / 1000.0], [m_, m_], color=C_EXP, lw=2.2)
    ax.plot(xk[KH] / 1000.0, Ek[KH], 'o', ms=11, mfc='none', mec='red', mew=1.8)
    ax.annotate(f'Streifen k = {KH}\nE = {Ek[KH]:.1f} GPa',
                (xk[KH] / 1000.0, Ek[KH]), textcoords='offset points',
                xytext=(8, 16), fontsize=7.5, color='red')
    h, l = ax.get_legend_handles_labels()
    h += [(band, line)]
    l += ['Experiment (DIC): Zonenmittel ± s\n(Auswertefenster ≫ Modellzone)']
    ax.legend(h, l, fontsize=7, loc='upper left', ncol=1,
              handler_map={tuple: HandlerTuple(ndivide=None)})
    ax.set_ylabel(r'$E(x)$ [GPa]')
    ax.set_ylim(140, 290)
    ax.grid(alpha=0.3)
    ax.set_title('(a) Lokaler E-Verlauf im simulierten Fenster', fontsize=10, loc='left')

    # --- Panel (b): die Vorfaktoren s(x) selbst ---------------------------
    ax = axs[1]
    xs_fine = np.linspace(0, Lx, 600)
    in_tr = (xs_fine >= XT[0]) & (xs_fine < XT[1])
    ax.plot(xs_fine / 1000.0, np.ones_like(xs_fine), '-', color=C_FE, lw=1.8)
    for tag, lab, sfun, col in SVAR:
        sv = np.where(in_tr, sfun(xs_fine), 1.0)
        ax.plot(xs_fine / 1000.0, sv, '-', color=col, lw=1.8)
    ax.set_ylabel('s(x)')
    ax.set_ylim(0.9, 1.65)
    ax.grid(alpha=0.3)
    ax.set_title('(b) Die angesetzten Vorfaktoren — wirken nur in der '
                 'Übergangszone', fontsize=10, loc='left')

    ax = axs[2]
    # Streifenmittel statt Summen: die Zellzahl kuerzt sich in E = sum/sum
    # ohnehin heraus, und die Streifen enthalten 29 bzw. 30 Spalten.
    ncell_k = np.array([(ib == k).sum() * ny for k in range(len(xk))])
    sig_k, eps_k = Ssum / ncell_k, Psum / ncell_k
    ax.plot(xk / 1000.0, sig_k / sig_k.mean(), 's-', color='#8172b3', lw=1.6, ms=4,
            label=r'Zähler $\langle\sigma_{xx}\rangle_k$ (auf Mittelwert normiert)')
    ax.plot(xk / 1000.0, eps_k / eps_k.mean(), 'd-', color='#937860', lw=1.6, ms=4,
            label=r'Nenner $\langle\varepsilon_{xx}\rangle_k$ (auf Mittelwert normiert)')
    ax.axvspan(xa / 1000.0, xb / 1000.0, color='red', alpha=0.12, lw=0)
    ax.set_xlabel('x [mm] im ROI (17-4PH → Übergang → 316L)')
    ax.set_ylabel('normiert auf Mittelwert')
    ax.legend(fontsize=7.5, loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_title('(c) Zerlegung für s(x) = 1: der Zähler ist konstant, die '
                 'gesamte Variation steckt im Nenner', fontsize=10, loc='left')
    for a_ in axs:
        for x_ in XT:
            a_.axvline(x_ / 1000.0, color='k', lw=0.9, ls='--', alpha=0.6)
    ax.set_xlim(0, Lx / 1000.0)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_E_profile.png'), dpi=150, bbox_inches='tight')
    print('wrote fig_E_profile.png')

    # --- Kennzahlen fuer den Berichtstext ----------------------------------
    cv_s = np.std([sig[:, ib == k].mean() for k in range(len(xk))]) / sig.mean()
    cv_e = np.std([eps[:, ib == k].mean() for k in range(len(xk))]) / eps.mean()
    ncell = int((ib == KH).sum() * ny)
    stats = {'bin_um': float(Lx / len(xk)), 'nbin': int(len(xk)),
             'cells_per_bin': ncell, 'k_example': KH,
             'x_example_um': [float(xa), float(xb)],
             'sum_sig_example_GPa': float(Ssum[KH]),
             'sum_eps_example': float(Psum[KH]),
             'E_example_GPa': float(Ek[KH]),
             'cv_strip_mean_sigma_pct': float(100 * cv_s),
             'cv_strip_mean_eps_pct': float(100 * cv_e),
             'E_profile_min_GPa': float(Ek.min()), 'E_profile_max_GPa': float(Ek.max())}
    json.dump(stats, open(os.path.join(HERE, 'profile_stats.json'), 'w'), indent=1)
    print('wrote profile_stats.json:', stats)

# ---- fig_dic_profile: Experiment ueber die ganze Probe ---------------------
DIC = [171.11, 169.2, 193.62, 195.56, 215.95, 237.23, 211.08,
       207.54, 209.63, 195.55, 182.73, 181.33, 180.86, 181.08]
fig, ax = plt.subplots(figsize=(7.6, 4.0))
ax.plot(range(1, 15), DIC, 'o-', color=C_EXP, lw=1.8, ms=5)
ax.axvspan(6.0 - 1.4, 6.0 + 1.4, color='0.55', alpha=0.35, lw=0)
ax.text(6.0, 244, 'im Modell erfasstes Fenster ≈ 2,8 mm\n(Lage angenommen)',
        fontsize=7.5, ha='center', va='bottom')
ax.annotate('316L', (1.3, 162), fontsize=8)
ax.annotate('Grenzfläche', (7.1, 232), fontsize=8)
ax.annotate('17-4PH', (11.3, 189), fontsize=8)
ax.set_xlabel('DIC-Messpunkt 1…14 entlang der Messstrecke (316L → 17-4PH)')
ax.set_ylabel('E [GPa]')
ax.set_ylim(150, 275)
ax.grid(alpha=0.3)
ax.set_title('Experimenteller E-Verlauf über die ganze Probe (DIC, Punktraster)',
             fontsize=10, loc='left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_dic_profile.png'), dpi=150, bbox_inches='tight')
print('wrote fig_dic_profile.png')

# ---- fig_grain_Ex_variants: Abb. 3 auch fuer die skalierten Varianten ------
variants = [('s(x) = 1 — reine Mikrostruktur', None)] + \
           [(lab, sfun) for _, lab, sfun, _ in SVAR]
xcell = (np.arange(nx) + 0.5) * step
in_tr = (zone == 1)
Emaps = []
for lab, sfun in variants:
    if sfun is None:
        Emaps.append((lab, Ex.copy()))
    else:
        sv = np.broadcast_to(np.asarray(sfun(xcell), float), (nx,))
        sc = np.where(in_tr, sv[None, :], 1.0)
        Emaps.append((lab, Ex * sc))
vmin = min(m.min() for _, m in Emaps)
vmax = max(m.max() for _, m in Emaps)
fig = plt.figure(figsize=(14, 7.4))
gs = fig.add_gridspec(3, 3, width_ratios=[2.3, 1, 0.035], hspace=0.42, wspace=0.25)
for i, (lab, Em) in enumerate(Emaps):
    ax = fig.add_subplot(gs[i, 0])
    im = ax.imshow(Em, extent=EXT, aspect='equal', cmap='plasma', vmin=vmin, vmax=vmax)
    for x_ in XT:
        ax.axvline(x_, color='w', lw=1.1, ls='--')
    ax.set_ylabel('y [µm]')
    ax.set_title(f'({"abc"[i]}) {lab}', fontsize=9.5, loc='left')
    if i == 2:
        ax.set_xlabel('x [µm] (Lastachse)')
    axh = fig.add_subplot(gs[i, 1])
    axh.hist(Em[in_tr].ravel(), bins=60, range=(80, 340), histtype='stepfilled',
             lw=1.4, density=True, color='#c44e52', alpha=0.75)
    axh.axvline(Em[in_tr].mean(), color='k', lw=1.2, ls='--')
    axh.set_xlim(80, 340)
    axh.set_ylabel('Dichte', fontsize=8)
    axh.tick_params(labelsize=7)
    axh.grid(alpha=0.3)
    axh.set_title(f'Übergangszone: Mittel {Em[in_tr].mean():.0f} GPa',
                  fontsize=8.5, loc='left')
    if i == 2:
        axh.set_xlabel(r'$E_x$ des Korns [GPa]', fontsize=8)
plt.colorbar(im, cax=fig.add_subplot(gs[:, 2]), label=r'$E_x$ des Korns [GPa]')
fig.savefig(os.path.join(HERE, 'fig_grain_Ex_variants.png'), dpi=150, bbox_inches='tight')
print('wrote fig_grain_Ex_variants.png')

# ---- fig_averaging: Streifenmittelung von Abb. 3 vs. FE-Loesung -----------
if FE is not None:
    S11 = np.linalg.inv(C)[:, :, 0, 0]
    Ea = np.array([Ex[:, ib == k].mean() for k in range(len(xk))])
    Eh = np.array([1.0 / (1.0 / Ex[:, ib == k]).mean() for k in range(len(xk))])
    Ev = np.array([1.0 / np.linalg.inv(C[:, ib == k].reshape(-1, 3, 3).mean(axis=0))[0, 0]
                   for k in range(len(xk))])
    fig, axs = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                            gridspec_kw={'height_ratios': [1.5, 1]})
    ax = axs[0]
    ax.plot(xk / 1000.0, Ev, '^-', color='#8172b3', lw=1.4, ms=4,
            label=r'Tensor-Voigt: $E_x$ aus $\langle\mathbf{C}\rangle_k$')
    ax.plot(xk / 1000.0, Ea, 's-', color='#dd8452', lw=1.6, ms=4,
            label=r'arithmetisch: $\langle E_x\rangle_k$  (Mittelung von Abb. 3)')
    ax.plot(xk / 1000.0, Ek, 'o-', color=C_FE, lw=2.0, ms=5,
            label='FE-Lösung (dolfinx)')
    ax.plot(xk / 1000.0, Eh, 'd-', color='#937860', lw=1.4, ms=4,
            label=r'harmonisch: $1/\langle 1/E_x\rangle_k$ = Tensor-Reuss')
    for x_ in XT:
        ax.axvline(x_ / 1000.0, color='k', lw=0.9, ls='--', alpha=0.6)
    ax.set_ylabel(r'$E(x)$ [GPa]')
    ax.set_ylim(140, 235)
    ax.legend(fontsize=7.5, loc='lower right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_title('(a) Streifenmittelung der kornweisen Moduln gegen die FE-Lösung',
                 fontsize=10, loc='left')
    ax = axs[1]
    ax.axhline(0, color='k', lw=0.8)
    ax.plot(xk / 1000.0, Ek - Ea, 's-', color='#dd8452', lw=1.6, ms=4,
            label='FE − arithmetisch')
    ax.plot(xk / 1000.0, Ek - Eh, 'd-', color='#937860', lw=1.4, ms=4,
            label='FE − harmonisch')
    ax.plot(xk / 1000.0, Ek - Ev, '^-', color='#8172b3', lw=1.4, ms=4,
            label='FE − Tensor-Voigt')
    for x_ in XT:
        ax.axvline(x_ / 1000.0, color='k', lw=0.9, ls='--', alpha=0.6)
    ax.set_xlabel('x [mm] im ROI (17-4PH → Übergang → 316L)')
    ax.set_ylabel('Abweichung [GPa]')
    ax.legend(fontsize=7.5, loc='lower right', ncol=3)
    ax.grid(alpha=0.3)
    ax.set_title('(b) Differenz zur FE-Lösung — im einkristallinen 316L-Bereich '
                 'exakt null', fontsize=10, loc='left')
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig_averaging.png'), dpi=150, bbox_inches='tight')
    print('wrote fig_averaging.png')
    # Relative Lage im Schrankenintervall (0 = Reuss, 1 = Voigt), nur Streifen
    # mit nennenswertem Kontrast - bei quasi-einkristallinen Streifen ist das
    # Intervall numerisch entartet.
    con = (Ev - Eh) > 5.0
    xi_fe = ((Ek - Eh) / (Ev - Eh))[con]
    xi_ar = ((Ea - Eh) / (Ev - Eh))[con]
    inside = int(np.sum((Ek[con] >= Eh[con] - 1e-6) & (Ek[con] <= Ev[con] + 1e-6)))
    avg = {'n_strips_with_contrast': int(con.sum()),
           'n_strips_FE_inside_bounds': inside,
           'xi_FE_mean': float(xi_fe.mean()), 'xi_FE_min': float(xi_fe.min()),
           'xi_FE_max': float(xi_fe.max()), 'xi_arith_mean': float(xi_ar.mean()),
           'mean_diff_arith': float((Ek - Ea).mean()),
           'max_abs_diff_arith': float(np.abs(Ek - Ea).max()),
           'mean_diff_harm': float((Ek - Eh).mean()),
           'mean_diff_voigt': float((Ek - Ev).mean()),
           'corr_arith': float(np.corrcoef(Ek, Ea)[0, 1]),
           'corr_harm': float(np.corrcoef(Ek, Eh)[0, 1]),
           'corr_voigt': float(np.corrcoef(Ek, Ev)[0, 1]),
           'zone_arith': {}, 'zone_harm': {}, 'zone_fe': {}}
    # Kennzahlen der Korn-Wechselwirkung (Bericht 3.7.4)
    sg, ep = FE['sig_xx'], FE['eps_xx']
    sm_, em_ = sg.mean(), ep.mean()
    band = (Ex > 200) & (Ex < 205)
    avg['interaction'] = {
        'sigma_mean_MPa': float(sm_ * 1e3),
        'sigma_min_MPa': float(sg.min() * 1e3), 'sigma_max_MPa': float(sg.max() * 1e3),
        'sigma_std_pct': float(100 * sg.std() / sm_),
        'sigma_p1_MPa': float(np.percentile(sg, 1) * 1e3),
        'sigma_p99_MPa': float(np.percentile(sg, 99) * 1e3),
        'concentration_max_over_mean': float(sg.max() / sm_),
        'corr_Ex_sigma': float(np.corrcoef(Ex.ravel(), sg.ravel())[0, 1]),
        'err_voigt_assumption_pct': float(100 * (np.abs(Ex * em_ - sg) / sm_).mean()),
        'err_reuss_assumption_pct': float(100 * (np.abs(sm_ - sg) / sm_).mean()),
        'band_200_205_n': int(band.sum()),
        'band_200_205_sigma_mean_MPa': float(sg[band].mean() * 1e3),
        'band_200_205_sigma_std_MPa': float(sg[band].std() * 1e3),
        'band_200_205_sigma_min_MPa': float(sg[band].min() * 1e3),
        'band_200_205_sigma_max_MPa': float(sg[band].max() * 1e3)}
    for z, nm in [(0, '17-4PH'), (1, 'trans'), (2, '316L')]:
        m = zone == z
        avg['zone_arith'][nm] = float(Ex[m].mean())
        avg['zone_harm'][nm] = float(1.0 / (1.0 / Ex[m]).mean())
        avg['zone_fe'][nm] = float(FE['sig_xx'][m].sum() / FE['eps_xx'][m].sum())
    json.dump(avg, open(os.path.join(HERE, 'averaging_stats.json'), 'w'), indent=1)
    print('wrote averaging_stats.json')


# ---- fig_mesh_zoom: reguläres Netz über der Kornstruktur -------------------
# Nur ein kleiner Ausschnitt - über dem ganzen ROI waeren die Elementkanten
# (3,371 µm) nicht mehr aufloesbar.
ZX0, ZY0, NCX, NCY = 250, 95, 42, 30            # Zellindizes + Ausschnittsgroesse
zx, zy = ZX0 * step, ZY0 * step
zw, zh = NCX * step, NCY * step
rec_full = np.asarray(Image.open(os.path.join(ROOT, 'micro_roi_ipf.png')))
sub_rec = rec_full[ZY0:ZY0 + NCY, ZX0:ZX0 + NCX]
sub_gid = gid[ZY0:ZY0 + NCY, ZX0:ZX0 + NCX]
ext_z = [zx, zx + zw, zy + zh, zy]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
if args.src and os.path.isfile(args.src):
    src = np.asarray(Image.open(args.src).convert('RGB'))
    H, W, _ = src.shape
    xcs = np.arange(ROI[0] + step / 2, ROI[2], step)[ZX0:ZX0 + NCX]
    ycs = np.arange(ROI[1] + step / 2, ROI[3], step)[ZY0:ZY0 + NCY]
    pxs = np.clip((xcs / S_UM_PER_PX).astype(int), 0, W - 1)
    pys = np.clip((ycs / S_UM_PER_PX).astype(int), 0, H - 1)
    ax.imshow(src[np.ix_(pys, pxs)], extent=ext_z, aspect='equal',
              interpolation='nearest')
ax.set_title('(a) EBSD-Originalscan des Ausschnitts', fontsize=10, loc='left')

ax2.imshow(sub_rec, extent=ext_z, aspect='equal', interpolation='nearest')
for i_ in range(NCX + 1):
    ax2.axvline(zx + i_ * step, color='k', lw=0.35, alpha=0.55)
for j_ in range(NCY + 1):
    ax2.axhline(zy + j_ * step, color='k', lw=0.35, alpha=0.55)
for j_ in range(NCY):
    for i_ in range(NCX):
        if i_ + 1 < NCX and sub_gid[j_, i_] != sub_gid[j_, i_ + 1]:
            ax2.plot([zx + (i_ + 1) * step] * 2,
                     [zy + j_ * step, zy + (j_ + 1) * step], color='k', lw=1.9)
        if j_ + 1 < NCY and sub_gid[j_, i_] != sub_gid[j_ + 1, i_]:
            ax2.plot([zx + i_ * step, zx + (i_ + 1) * step],
                     [zy + (j_ + 1) * step] * 2, color='k', lw=1.9)
ax2.plot([zx + 1.5 * step, zx + 2.5 * step], [zy + zh - 1.3 * step] * 2, 'k-', lw=3)
ax2.annotate(f'{step:.2f} µm = 1 Element', (zx + 2.0 * step, zy + zh - 1.9 * step),
             ha='left', va='bottom', fontsize=8,
             bbox=dict(fc='white', alpha=0.9, ec='none', pad=1.5))
ax2.set_title('(b) Modelleingang: ein Q1-Element je EBSD-Pixel\n'
              'dünn = Elementkanten, dick = Korngrenzen (treppenförmig)',
              fontsize=10, loc='left')
for a_ in (ax, ax2):
    a_.set_xlabel('x [µm]')
    a_.set_ylabel('y [µm]')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_mesh_zoom.png'), dpi=170, bbox_inches='tight')
print('wrote fig_mesh_zoom.png')

# ---- fig_strips: Skizze der Streifenaufteilung -----------------------------
NBIN = 28
edg = np.linspace(0.0, Lx, NBIN + 1)
KH2 = 12
fig = plt.figure(figsize=(13, 6.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], width_ratios=[1, 0.72],
                      hspace=0.62, wspace=0.18)

# (a) ROI mit allen Streifen
ax = fig.add_subplot(gs[0, :])
for z0, z1, col, lab in [(0, XT[0], '#c9c9e6', '17-4PH'),
                         (XT[0], XT[1], '#e8d5c2', 'Übergang'),
                         (XT[1], Lx, '#cfe0cf', '316L')]:
    ax.add_patch(plt.Rectangle((z0, 0), z1 - z0, Ly, fc=col, ec='none'))
    ax.text((z0 + z1) / 2, Ly * 0.5, lab, ha='center', va='center', fontsize=10,
            bbox=dict(fc='white', alpha=0.8, ec='none', pad=2))
for e in edg:
    ax.plot([e, e], [0, Ly], color='0.45', lw=0.6)
ax.add_patch(plt.Rectangle((edg[KH2], 0), edg[KH2 + 1] - edg[KH2], Ly,
                           fc='#c44e52', ec='red', lw=1.8, alpha=0.5))
ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, ec='k', lw=1.4))
# Breitenmass des Streifens
ax.annotate('', xy=(edg[KH2], -90), xytext=(edg[KH2 + 1], -90),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.3))
ax.text((edg[KH2] + edg[KH2 + 1]) / 2, -160,
        f'Streifen k = {KH2}:  {Lx/NBIN:.2f} µm breit',
        ha='center', va='top', fontsize=9, color='red')
# Hoehe
ax.annotate('', xy=(Lx + 80, 0), xytext=(Lx + 80, Ly),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.1))
ax.text(Lx + 120, Ly / 2, f'{Ly:.0f} µm\n263 Zellen', va='center', fontsize=9)
# Gesamtbreite
ax.annotate('', xy=(0, Ly + 170), xytext=(Lx, Ly + 170),
            arrowprops=dict(arrowstyle='<->', color='k', lw=1.1))
ax.text(Lx / 2, Ly + 215, f'{Lx:.0f} µm  =  {NBIN} Streifen à {Lx/NBIN:.2f} µm',
        ha='center', va='bottom', fontsize=9)
# Lastrichtung
ax.annotate('', xy=(Lx * 0.16, -350), xytext=(0, -350),
            arrowprops=dict(arrowstyle='->', lw=2))
ax.text(Lx * 0.17, -350, 'Lastrichtung x', va='center', fontsize=9)
ax.set_xlim(-90, Lx + 480)
ax.set_ylim(-470, Ly + 330)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(a) Streifenaufteilung des ROI entlang der Lastachse — jeder Streifen '
             'reicht über die volle Höhe', fontsize=10, loc='left')

# (b) Zoom in einen Streifen
ax = fig.add_subplot(gs[1, 0])
NSX, NSY = 29, 10
for i_ in range(NSX + 1):
    ax.plot([i_, i_], [0, NSY], color='0.45', lw=0.6)
for j_ in range(NSY + 1):
    ax.plot([0, NSX], [j_, j_], color='0.45', lw=0.6)
ax.add_patch(plt.Rectangle((0, 0), NSX, NSY, fc='#c44e52', ec='red', lw=2.0, alpha=0.14))
ax.annotate('', xy=(0, NSY + 1.1), xytext=(NSX, NSY + 1.1),
            arrowprops=dict(arrowstyle='<->', lw=1.1))
ax.text(NSX / 2, NSY + 1.7, '29 Spalten', ha='center', fontsize=9)
ax.annotate('', xy=(-1.3, 0), xytext=(-1.3, NSY),
            arrowprops=dict(arrowstyle='<->', lw=1.1))
ax.text(-2.2, NSY / 2, '263\nZeilen', ha='right', va='center', fontsize=9)
ax.text(NSX / 2, NSY / 2, '⋮', ha='center', va='center', fontsize=26, color='0.3')
ax.text(NSX / 2, -1.9, '29 × 263 = 7627 Elemente', ha='center', fontsize=9.5)
ax.text(NSX / 2, -3.4, r'je Element ein Wertepaar $(\sigma_{xx}^{(e)},\ '
        r'\varepsilon_{xx}^{(e)})$', ha='center', fontsize=9)
ax.set_xlim(-7.5, NSX + 1)
ax.set_ylim(-5.2, NSY + 3)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('(b) Ein Streifen von innen (Höhe verkürzt dargestellt)',
             fontsize=10, loc='left')

# (c) Auswertevorschrift
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
ax.text(0.02, 1.00, 'Auswertung je Streifen', fontsize=10.5, va='top', weight='bold')
ax.text(0.02, 0.82, r'$E(x_k)\;=\;\dfrac{\Sigma_{e\in k}\,\sigma_{xx}^{(e)}}'
        r'{\Sigma_{e\in k}\,\varepsilon_{xx}^{(e)}}$', fontsize=16, va='top')
ax.text(0.02, 0.40, 'Zähler:  durch das Kräftegleichgewicht\n'
        '             in allen Streifen gleich\n\n'
        'Nenner:  trägt die gesamte\n'
        '             Ortsabhängigkeit von E(x)', fontsize=9, va='top', linespacing=1.5)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
fig.savefig(os.path.join(HERE, 'fig_strips.png'), dpi=170, bbox_inches='tight')
print('wrote fig_strips.png')

# ---- fig_ferrite: Ferritgehalt und lokaler E-Verlauf, gleiche Probe --------
# Beide Datensaetze stammen von derselben Probe (MHo1030_A9D_5) und denselben
# 14 Messpunkten - der Vergleich braucht daher keine Registrierungsannahme.
FERRIT = [4.03, 4.50, 5.53, 7.10, 31.83, 38.70, 40.70, 43.00,
          44.27, 45.03, 44.93, 45.27, 45.43, 44.90]
E_DIC = [171.11, 169.2, 193.62, 195.56, 215.95, 237.23, 211.08,
         207.54, 209.63, 195.55, 182.73, 181.33, 180.86, 181.08]
pts = np.arange(1, 15)
fig, axs = plt.subplots(2, 1, figsize=(8.4, 5.8), sharex=True,
                        gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.18})
ax = axs[0]
ax.plot(pts, E_DIC, 'o-', color=C_EXP, lw=1.9, ms=5)
ax.axvspan(4.5, 6.5, color='0.6', alpha=0.28, lw=0)
ax.annotate(f'Maximum {max(E_DIC):.0f} GPa', (6, max(E_DIC)),
            textcoords='offset points', xytext=(14, -12), fontsize=8.5, color=C_EXP)
ax.set_ylim(160, 250)
ax.set_ylabel('E lokal [GPa]  (DIC)')
ax.grid(alpha=0.3)
ax.set_title('(a) Lokaler E-Modul, Probe MHo1030_A9D_5', fontsize=10, loc='left')
ax = axs[1]
ax.plot(pts, FERRIT, 's-', color='#937860', lw=1.9, ms=5, label='Ferritgehalt')
ax.axvspan(4.5, 6.5, color='0.6', alpha=0.28, lw=0)
ax.set_ylabel('Ferritgehalt [%]')
ax.set_xlabel('Messpunkt 1…14 entlang der Messstrecke (316L → 17-4PH)')
ax.grid(alpha=0.3)
axg = ax.twinx()
axg.plot(pts, np.abs(np.gradient(FERRIT)), '^--', color='#c44e52', lw=1.3, ms=4)
axg.set_ylabel('|Gradient| [%/Punkt]', color='#c44e52', fontsize=9)
axg.tick_params(axis='y', colors='#c44e52', labelsize=8)
ax.set_title('(b) Ferritgehalt (Ferritscope) und dessen Gradient — dieselbe Probe, '
             'dieselben Messpunkte', fontsize=10, loc='left')
fig.savefig(os.path.join(HERE, 'fig_ferrite.png'), dpi=170, bbox_inches='tight')
print('wrote fig_ferrite.png')
