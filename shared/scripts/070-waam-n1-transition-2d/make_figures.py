#!/usr/bin/env python3
"""
Result figures for the 2D transition-zone model.

Data source: EXCLUSIVELY the dolfinx FE results written by
`solve_plane_stress.py` — `E_<tag>.json` (moduli, E(x) profile) and
`fields_<tag>.npz` (cell fields on the microstructure grid). No other solver
feeds these figures; if a file is missing the script says which dolfinx run to
start instead of silently substituting anything.

Panels
  (a) EBSD-reconstructed microstructure (IPF-Z) of the ROI
  (b) E_x per cell - the directional modulus of each GRAIN's own rotated,
      plane-stress-condensed tensor. This is the field that makes the
      per-grain stiffness visible (`phase` has only 2 values by design).
  (c) sigma_xx at eps_xx = 1e-3
  (d) per-zone local modulus: experiment (DIC) vs. model
  (e) local E(x) profile of the MODEL over the SIMULATED window only. The
      experimental DIC value of each zone is drawn over that zone's own
      x-range as a solid line (zone mean) inside a pale band (+- std); both
      share one combined legend key.
  (f) full-gauge DIC profile (whole specimen, separate point-grid evaluation)
      with the simulated window marked. NOTE: (e) is NOT a zoom of (f) - the
      blue levels in (e) are the zone means of the tensile report (area
      averages over evaluation windows several mm wide), while (f) is the
      14-point local profile of one specimen. Two different DIC evaluations.

Usage: python3 make_figures.py
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

C_EXP, C_S1, C_S133, C_GAUSS = '#4c72b0', '#c44e52', '#dd8452', '#55a868'

ROI = (30., 1339., 2811., 2225.)     # ROI in map coordinates [um]
X_TRANS_MAP = (666.6, 1494.5)        # transition zone, map coordinates [um]
X_TRANS = tuple(x - ROI[0] for x in X_TRANS_MAP)   # ... in ROI coordinates

# experimental (DIC) zone moduli, Kennwerte_Zugversuche_WAAM_N=1
EXP_ZONE = {'17-4PH': (201.7, 2.0), 'transition': (232.4, 13.0),
            '316L': (162.7, 2.0)}
# full-gauge local DIC profile, specimen MHo1030_A9D_5 (316L -> 17-4PH)
DIC = [171.11, 169.2, 193.62, 195.56, 215.95, 237.23, 211.08,
       207.54, 209.63, 195.55, 182.73, 181.33, 180.86, 181.08]


def load(tag, required=True):
    """Load a dolfinx FE result. Only `E_<tag>.json` written by
    solve_plane_stress.py is accepted - no other solver feeds these figures."""
    p = os.path.join(HERE, f'E_{tag}.json')
    if not os.path.isfile(p):
        if not required:
            return None
        raise SystemExit(
            f'E_{tag}.json fehlt.\n'
            f'Im dolfinx-Container rechnen, z.B.:\n'
            f'  python3 solve_plane_stress.py --micro micro_roi.npz '
            f'--tag {tag} --sfun "1.0"')
    d = json.load(open(p))
    d['_source'] = 'dolfinx'
    return d


res = load('roi')
res_s = load('roi_s133', required=False)
res_g = load('roi_gauss', required=False)
SRC = 'dolfinx'

fp = os.path.join(HERE, 'fields_roi.npz')
if not os.path.isfile(fp):
    raise SystemExit('fields_roi.npz fehlt - solve_plane_stress.py erneut laufen '
                     'lassen (schreibt die Gitterfelder mit).')
F = np.load(fp)
s_xx = F['sig_xx'] * 1000.0          # GPa -> MPa, Zeile 0 = oben in der Map
Ex = F['E_x']
Lx, Ly = res['Lx_um'], res['Ly_um']
ipf = np.asarray(Image.open(os.path.join(HERE, 'micro_roi_ipf.png')))
EXT = [0, Lx, Ly, 0]

fig = plt.figure(figsize=(16.5, 7.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.15], hspace=0.45, wspace=1.0)
ax_a, ax_b, ax_c = (fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
                    fig.add_subplot(gs[0, 4:6]))
ax_d, ax_e, ax_f = (fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
                    fig.add_subplot(gs[1, 4:6]))


def zone_lines(ax, color='k'):
    for xb in X_TRANS:
        ax.axvline(xb, color=color, lw=1.0, ls='--')


# ---- (a) microstructure ----------------------------------------------------
ax = ax_a
ax.imshow(ipf, extent=EXT, aspect='equal')
zone_lines(ax)
for fx, lab in [(0.115, '17-4PH'), (0.38, 'Übergang'), (0.76, '316L')]:
    ax.text(fx, 0.06, lab, transform=ax.transAxes, ha='center', fontsize=8,
            bbox=dict(fc='white', alpha=0.85, ec='none', pad=1.5))
ax.set_xlabel('x [µm] (Last-/Aufbauachse)')
ax.set_ylabel('y [µm]')
ax.set_title('(a) EBSD-Mikrostruktur (IPF-Z)', fontsize=10, loc='left')

# ---- (b) per-grain directional modulus -------------------------------------
ax = ax_b
if Ex is not None:
    im = ax.imshow(Ex, extent=EXT, aspect='equal', cmap='plasma')
    plt.colorbar(im, ax=ax, label=r'$E_x$ des Korns [GPa]', fraction=0.03, pad=0.02)
else:
    ax.text(0.5, 0.5, "Feld 'E_x' fehlt", ha='center', va='center',
            transform=ax.transAxes, fontsize=8)
zone_lines(ax, 'w')
ax.set_xlabel('x [µm]')
ax.set_ylabel('y [µm]')
ax.set_title(r'(b) $E_x$ je Korn aus dessen eigenem Tensor', fontsize=10, loc='left')

# ---- (c) sigma_xx field ----------------------------------------------------
ax = ax_c
im = ax.imshow(s_xx, extent=EXT, aspect='equal', cmap='viridis',
               vmin=np.percentile(s_xx, 2), vmax=np.percentile(s_xx, 98))
zone_lines(ax, 'w')
plt.colorbar(im, ax=ax, label=r'$\sigma_{xx}$ [MPa]', fraction=0.03, pad=0.02)
ax.set_xlabel('x [µm]')
ax.set_ylabel('y [µm]')
ax.set_title(r'(c) $\sigma_{xx}$ bei $\varepsilon_{xx}=10^{-3}$, s(x)=1',
             fontsize=10, loc='left')

# ---- (d) per-zone moduli ---------------------------------------------------
ax = ax_d
zones = ['17-4PH', 'transition', '316L']
zlabel = ['17-4PH', 'Übergang/\nGrenzfläche', '316L']
exp = [EXP_ZONE[z][0] for z in zones]
err = [EXP_ZONE[z][1] for z in zones]
series = [('Experiment (DIC)', exp, C_EXP, err),
          ('Modell s(x)=1 (dolfinx)',
           [res[f'zone_{z}']['E_local_GPa'] for z in zones], C_S1, None)]
if res_s:
    series.append(('Modell s=1,33 (dolfinx)',
                   [res_s[f'zone_{z}']['E_local_GPa'] for z in zones], C_S133, None))
if res_g:
    series.append(('Modell s(x) Gauß (dolfinx)',
                   [res_g[f'zone_{z}']['E_local_GPa'] for z in zones], C_GAUSS, None))
x = np.arange(3)
w = 0.8 / len(series)
for i, (lab, vals, col, yerr) in enumerate(series):
    xs = x + (i - (len(series) - 1) / 2) * w
    ax.bar(xs, vals, w * 0.9, yerr=yerr, capsize=3, color=col, label=lab)
    off = np.asarray(yerr) if yerr is not None else np.zeros(len(vals))
    for xi, v, o in zip(xs, vals, off):
        ax.text(xi, v + o + 7, f'{v:.0f}', ha='center', fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(zlabel)
ax.set_ylabel(r'$E_x$ lokal [GPa]')
ax.set_ylim(0, 335)
ax.legend(fontsize=7, loc='upper center', ncol=2, framealpha=0.95)
ax.grid(axis='y', alpha=0.3)
ax.set_title('(d) Steifigkeit je Zone: Experiment vs. Modell',
             fontsize=10, loc='left')

# ---- (e) model E(x) over the SIMULATED window ------------------------------
ax = ax_e
curves = [('Modell s(x)=1 (dolfinx)', res, C_S1)]
if res_s:
    curves.append(('Modell s=1,33 (dolfinx)', res_s, C_S133))
if res_g:
    curves.append(('Modell s(x) Gauß (dolfinx)', res_g, C_GAUSS))
for lab, r, col in curves:
    p = r.get('E_profile')
    if p:
        ax.plot(np.array(p['x_um']) / 1000.0, p['E_GPa'], '-', color=col,
                lw=1.8, label=lab)
seg = {'17-4PH': (0.0, X_TRANS[0]), 'transition': X_TRANS, '316L': (X_TRANS[1], Lx)}
band = line = None
for z, (xa, xb) in seg.items():
    m, s = EXP_ZONE[z]
    band = ax.fill_between([xa / 1000.0, xb / 1000.0], m - s, m + s,
                           color=C_EXP, alpha=0.25, lw=0)
    line, = ax.plot([xa / 1000.0, xb / 1000.0], [m, m], color=C_EXP, lw=2.2)
zb = None
for xb in X_TRANS:
    zb = ax.axvline(xb / 1000.0, color='k', lw=0.9, ls='--', alpha=0.6)
ax.set_xlabel('x [mm] im ROI (17-4PH → Übergang → 316L)')
ax.set_ylabel(r'$E_x$ lokal [GPa]')
ax.set_xlim(0, Lx / 1000.0)
ax.set_ylim(140, 325)
# legend: the solid blue line is the zone MEAN, the pale band its +-s -> one
# combined key, so no unexplained line is left in the plot
handles, labels = ax.get_legend_handles_labels()
handles += [(band, line), zb]
labels += ['Experiment (DIC): Zonenmittel ± s\n(Auswertefenster ≫ Modellzone)',
           'Zonengrenzen']
ax.legend(handles, labels, fontsize=7, loc='upper left', framealpha=0.95,
          handler_map={tuple: HandlerTuple(ndivide=None)})
ax.grid(alpha=0.3)
ax.set_title('(e) E-Verlauf im simulierten Fenster (≈2,8 mm)',
             fontsize=10, loc='left')

# ---- (f) full-gauge DIC profile --------------------------------------------
ax = ax_f
ax.plot(range(1, 15), DIC, 'o-', color=C_EXP, lw=1.8, ms=4.5,
        label='DIC lokal (MHo1030, bis 1,5 % Dehnung)')
# 2.8 mm model window at the 1 mm spacing of the inner DIC points; its
# POSITION is an assumption (peak = interface), the registration EBSD-ROI <->
# point index is not known.
ax.axvspan(6.0 - 1.4, 6.0 + 1.4, color='0.55', alpha=0.35, lw=0)
ax.text(6.0, 252, 'Modellfenster ≈2,8 mm\n(Lage angenommen)', fontsize=7,
        ha='center', va='bottom')
ax.annotate('316L', (1.4, 162), fontsize=8)
ax.annotate('Grenzfläche', (7.0, 232), fontsize=8)
ax.annotate('17-4PH', (11.3, 190), fontsize=8)
ax.set_xlabel('DIC-Messpunkt 1…14 (ganze Messstrecke)')
ax.set_ylabel('E [GPa]')
ax.set_ylim(150, 285)
ax.legend(fontsize=7, loc='upper right')
ax.grid(alpha=0.3)
ax.set_title('(f) Experiment: E-Verlauf über die ganze Probe',
             fontsize=10, loc='left')
ax.text(0.5, -0.30, 'separate DIC-Auswertung (Punktraster); (e) ist kein '
        'Ausschnitt hiervon', transform=ax.transAxes, ha='center',
        fontsize=6.5, style='italic', color='0.35')

out = os.path.join(HERE, 'fig_roi_overview.png')
fig.savefig(out, dpi=170, bbox_inches='tight')
print(f'wrote {out}  (Quelle Modellwerte: {SRC})')
