#!/usr/bin/env python3
"""
Abbildungen der Höhenstudie (Kapitel 3 des Berichts).

  fig_study_map.png       Karte mit allen Auswertefenstern + Lage der
                          Zonengrenzen über die Probenhöhe
  fig_study_bounds.png    Voigt-/Reuss-Bereich der Übergangszone je Fenster
                          gegen den Messwert
  fig_study_profiles.png  Streifenprofile E(x) aller Fenster

Quelle: `study_cases.json` + `study_stats.json` (erzeugt von `study_rois.py`
und `study_bounds.py`) sowie die `micro_<tag>.npz` — das sind Schranken und der
in Anhang C validierte arithmetische Schaetzer, kein Solver. Liegen zusaetzlich
`E_<tag>.json` / `fields_<tag>.npz` aus `run_study.sh` vor, wird die
dolfinx-Loesung mit eingezeichnet (Raute bzw. schwarze Kurve).

Aufruf: python3 report/make_study_figs.py --src <...Uebergangsbereich.bmp>
"""
import argparse, json, os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

C_EXP, C_FE, C_V, C_R = '#4c72b0', '#c44e52', '#8172b3', '#dd8452'
E_EXP_TR, E_EXP_TR_SD = 232.4, 13.0
S_UM_PER_PX = 3.371
ORDER = ['band1', 'band2', 'band3', 'band4', 'roi', 'full']
NICE = {'band1': 'Band 1\n(oben)', 'band2': 'Band 2', 'band3': 'Band 3',
        'band4': 'Band 4\n(unten)', 'roi': 'Referenz-\nband',
        'full': 'volle\nHöhe'}
FLAT = {'band1': 'Band 1 (oben)', 'band2': 'Band 2', 'band3': 'Band 3',
        'band4': 'Band 4 (unten)', 'roi': 'Referenzband', 'full': 'volle Höhe'}

ap = argparse.ArgumentParser()
ap.add_argument('--src', default=None, help='WAAM_N=1_A12D_Uebergangsbereich.bmp')
args = ap.parse_args()

cases = {c['tag']: c for c in json.load(open(os.path.join(ROOT, 'study_cases.json')))}
stats = json.load(open(os.path.join(ROOT, 'study_stats.json')))
S = {c['tag']: c for c in stats['cases']}


def fe_result(tag):
    """dolfinx-Ergebnis des Falls, falls vorhanden: Zonenmoduln + E(x)-Profil.

    Liest ausschliesslich `E_<tag>.json` / `fields_<tag>.npz` aus dem
    dolfinx-Lauf (`run_study.sh`). Ohne diese Dateien -> None, und die
    Abbildungen zeigen nur Schranken und Schaetzer.
    """
    ej = os.path.join(ROOT, f'E_{tag}.json')
    if not os.path.isfile(ej):
        return None
    E = json.load(open(ej))
    out = {'zones': {n: E.get(f'zone_{k}', {}).get('E_local_GPa')
                     for n, k in [('17-4PH', '17-4PH'), ('Übergang', 'transition'),
                                  ('316L', '316L')]},
           'E_apparent': E.get('E_apparent_GPa')}
    fp = os.path.join(ROOT, f'fields_{tag}.npz')
    if os.path.isfile(fp):
        F = np.load(fp)
        sig, eps = F['sig_xx'], F['eps_xx']
        d = np.load(os.path.join(ROOT, f'micro_{tag}.npz'))
        xc = d['x_um']; step = json.loads(str(d['meta']))['step_um']
        nx = sig.shape[1]
        k = max(1, int(round(nx * step / 100.0)))
        edges = np.linspace(0, nx, k + 1).astype(int)
        out['profile'] = [(float(0.5 * (xc[a] + xc[b - 1])),
                           float(sig[:, a:b].sum() / eps[:, a:b].sum()))
                          for a, b in zip(edges[:-1], edges[1:])]
    return out


FE = {t: fe_result(t) for t in ORDER}
if any(FE.values()):
    print('FE-Ergebnisse gefunden für: '
          + ', '.join(t for t, v in FE.items() if v))
else:
    print('keine FE-Ergebnisse vorhanden — Abbildungen zeigen Schranken '
          'und den arithmetischen Schätzer')


# ---- fig_study_map --------------------------------------------------------
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1])
ax = fig.add_subplot(gs[0, 0])
if args.src and os.path.isfile(args.src):
    im = np.asarray(Image.open(args.src).convert('RGB'))
    H, W, _ = im.shape
    ax.imshow(im, extent=[0, W * S_UM_PER_PX, H * S_UM_PER_PX, 0],
              interpolation='nearest')
    ax.set_xlim(0, W * S_UM_PER_PX); ax.set_ylim(H * S_UM_PER_PX, 0)
for tag in ['band1', 'band2', 'band3', 'band4']:
    c = cases[tag]
    ax.add_patch(Rectangle((30, c['y0']), 2781, c['y1'] - c['y0'],
                           fill=False, ec='k', lw=2.0))
    ax.text(60, 0.5 * (c['y0'] + c['y1']), tag.replace('band', 'Band '),
            fontsize=9, va='center', color='k',
            bbox=dict(fc='white', alpha=0.85, ec='none', pad=1.5))
    ax.plot([c['zones'][0]] * 2, [c['y0'], c['y1']], color='lime', lw=2.2)
    ax.plot([c['zones'][1]] * 2, [c['y0'], c['y1']], color='lime', lw=2.2)
r = cases['roi']
ax.add_patch(Rectangle((30, r['y0']), 2781, r['y1'] - r['y0'],
                       fill=False, ec='w', lw=2.6, ls='--'))
ax.text(2760, 0.5 * (r['y0'] + r['y1']), 'Referenzband', fontsize=9, ha='right',
        va='center', bbox=dict(fc='white', alpha=0.85, ec='none', pad=1.5))
ax.set_xlabel('x [µm] (Lastachse)'); ax.set_ylabel('y [µm]')
ax.set_title('(a) Auswertefenster im EBSD-Scan; grün: Zonengrenzen je Band',
             fontsize=10, loc='left')

ax2 = fig.add_subplot(gs[0, 1])
for tag in ORDER[:4]:
    c = cases[tag]
    ym = 0.5 * (c['y0'] + c['y1'])
    ax2.plot([c['zones'][0], c['zones'][1]], [ym, ym], color=C_V, lw=7,
             alpha=0.55, solid_capstyle='butt')
    ax2.plot(c['zones'][0], ym, 'o', color='k', ms=5)
    ax2.plot(c['zones'][1], ym, 'o', color='k', ms=5, mfc='w')
ym = 0.5 * (r['y0'] + r['y1'])
ax2.plot([r['zones'][0], r['zones'][1]], [ym, ym], color=C_EXP, lw=7, alpha=0.55,
         solid_capstyle='butt', label='Referenzband (Markierung)')
d = cases['roi']['zones_detected']
ax2.plot(d, [ym - 120] * 2, 'x', color=C_FE, ms=7, mew=2)
ax2.plot([d[0], d[1]], [ym - 120] * 2, color=C_FE, lw=1.2, ls=':',
         label='dieselbe Vorschrift auf das Referenzband')
ax2.axvline(cases['full']['zones'][0], color='0.5', ls='--', lw=1)
ax2.axvline(cases['full']['zones'][1], color='0.5', ls='--', lw=1)
ax2.text(0.5 * sum(cases['full']['zones']), 60, 'volle Höhe', fontsize=8,
         ha='center', color='0.4')
ax2.set_ylim(3698, 0); ax2.set_xlim(0, 2811)
ax2.set_xlabel('x [µm]'); ax2.set_ylabel('y [µm]')
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8, loc='lower right')
ax2.set_title('(b) Lage und Breite der Übergangszone über die Probenhöhe\n'
              '(aus BCC-Anteil und Korngröße bestimmt)', fontsize=10, loc='left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_study_map.png'), dpi=150, bbox_inches='tight')
print('wrote fig_study_map.png')


# ---- fig_study_bounds -----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
for ax, zname in zip(axes, ['Übergang', '316L']):
    xs = np.arange(len(ORDER))
    for i, tag in enumerate(ORDER):
        z = S[tag]['zones'].get(zname)
        if z is None:
            continue
        ax.plot([i, i], [z['E_reuss'], z['E_voigt']], color=C_V, lw=11,
                alpha=0.5, solid_capstyle='butt',
                label='Voigt-/Reuss-Bereich' if i == 0 else None)
        ax.plot(i, z['E_arith'], 'o', color=C_FE, ms=8, zorder=5, mfc='none',
                mew=2, label='arithm. Schätzer' if i == 0 else None)
        fe = (FE.get(tag) or {}).get('zones', {}).get(zname)
        if fe:
            ax.plot(i, fe, 'D', color=C_FE, ms=7, zorder=6,
                    label='FE (dolfinx)' if i == 0 else None)
        ax.text(i + 0.16, z['E_voigt'], f"{z['E_voigt']:.0f}", fontsize=8,
                va='center', color=C_V)
    e, sd = ((E_EXP_TR, E_EXP_TR_SD) if zname == 'Übergang' else (162.7, 2.0))
    ax.axhspan(e - sd, e + sd, color=C_EXP, alpha=0.18)
    ax.axhline(e, color=C_EXP, lw=2, label=f'Messwert {e:.1f} GPa')
    ax.set_xticks(xs); ax.set_xticklabels([NICE[t] for t in ORDER], fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'{"(a)" if zname == "Übergang" else "(b)"} Zone '
                 f'{"Übergang" if zname == "Übergang" else "316L"}',
                 fontsize=11, loc='left')
axes[0].set_ylabel('$E_x$ [GPa]')
axes[0].legend(fontsize=9, loc='lower left')
axes[1].legend(fontsize=9, loc='lower left')
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_study_bounds.png'), dpi=150, bbox_inches='tight')
print('wrote fig_study_bounds.png')


# ---- fig_study_profiles ---------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), sharex=True, sharey=True)
for ax, tag in zip(axes.ravel(), ORDER):
    p = S[tag]['profile']
    x = np.array([q['x_um'] for q in p]) / 1000.0
    a = np.array([q['E_arith'] for q in p])
    lo = np.array([q['E_reuss'] for q in p])
    hi = np.array([q['E_voigt'] for q in p])
    ax.fill_between(x, lo, hi, color=C_V, alpha=0.28, lw=0,
                    label='Voigt-/Reuss-Bereich')
    ax.plot(x, a, '-o', color=C_FE, ms=3, lw=1.6, label='arithm. Schätzer')
    fp = (FE.get(tag) or {}).get('profile')
    if fp:
        ax.plot([q[0] / 1000.0 for q in fp], [q[1] for q in fp], '-', color='k',
                lw=1.8, label='FE (dolfinx)')
    ax.axhline(E_EXP_TR, color=C_EXP, lw=1.6, ls='-')
    ax.axhspan(E_EXP_TR - E_EXP_TR_SD, E_EXP_TR + E_EXP_TR_SD, color=C_EXP,
               alpha=0.15)
    z = S[tag]['zones_um']
    for zz in z:
        ax.axvline(zz / 1000.0, color='0.35', ls='--', lw=1)
    ax.set_title(f"{FLAT[tag]} — "
                 f"{S[tag]['ny']}×{S[tag]['nx']} Zellen, "
                 f"{S[tag]['n_grains']} Körner", fontsize=9, loc='left')
    ax.grid(alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel('x [mm] (Lastachse)')
for ax in axes[:, 0]:
    ax.set_ylabel('$E_x$ [GPa]')
axes[0, 0].legend(fontsize=8, loc='lower right')
axes[0, 0].text(0.02, 0.93, 'blau: Messwert Grenzfläche 232,4 GPa',
                transform=axes[0, 0].transAxes, fontsize=8, color=C_EXP)
fig.tight_layout()
fig.savefig(os.path.join(HERE, 'fig_study_profiles.png'), dpi=150,
            bbox_inches='tight')
print('wrote fig_study_profiles.png')
