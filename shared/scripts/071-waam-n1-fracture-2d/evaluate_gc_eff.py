#!/usr/bin/env python3
"""
Post-processing for 071: turn the J(t) history of one or more runs into an
EFFECTIVE FRACTURE TOUGHNESS Gc_eff and compare crack growth along the build
direction with crack growth transverse to it.

Idea (same as 067): under the surfing boundary condition the crack settles
into quasi-steady growth. The far-field J then fluctuates around a plateau -
the fluctuations ARE the microstructure (each grain the tip enters changes the
local driving force). The plateau value is the effective, microstructure-
averaged energy release rate needed to keep the crack running:

    Gc_eff = < J_x >   over the steady window
    K_eff  = sqrt(E' * Gc_eff)

Because Gc is constant in the model, any difference between Gc_eff and Gc, and
any difference between the two crack directions, comes purely from the elastic
heterogeneity/anisotropy of the grain structure.

Input:  run_fracture_simulation_<tag>_graphs.txt + run_meta_<tag>.json
Output: gc_eff_<tag>.json, gc_eff_summary.json, fig_J_vs_crack_tip.png

    python3 evaluate_gc_eff.py long trans
    python3 evaluate_gc_eff.py --glob 'run_fracture_simulation_*_graphs.txt'
"""
import argparse
import glob as globmod
import json
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

COLUMNS = ['t', 'Jx', 'Jy', 'x_crack_tip', 'x_K_field', 'Rx_top', 'Ry_top',
           'dW', 'W', 'A_surf', 'dt', 'E_el', 'E_surf', 'E_total',
           'newton_iters', 's_min']


def read_graphs(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            rows.append([float(v) for v in line.split()])
    if not rows:
        raise SystemExit(f'{path}: no data rows')
    a = np.asarray(rows)
    ncol = min(a.shape[1], len(COLUMNS))
    return {COLUMNS[k]: a[:, k] for k in range(ncol)}


def crack_grew(d, meta, min_growth_um=None):
    """Did the crack tip actually advance? If not, no plateau exists and any
    Gc_eff would be meaningless - say so instead of reporting nan."""
    x = d['x_crack_tip']
    x = x[np.isfinite(x)]
    if x.size == 0:
        return False, 0.0
    growth = float(np.nanmax(x) - np.nanmin(x))
    if min_growth_um is None:
        # Below ~2*epsilon the tip has not propagated, it has only formed its
        # regularised profile. A 1e-6 threshold would call that "growth" and
        # then produce an empty steady window further down.
        min_growth_um = 2.0 * float(meta.get('epsilon_um', 0.0) or 0.0) or 1e-6
    return growth > min_growth_um, growth


def steady_window(d, meta, frac_lo=0.35):
    """Indices of quasi-steady growth, defined on the CRACK TIP POSITION (not
    on time, so runs with different time stepping stay comparable).

    Upper end: the crack tip position where the MICROSTRUCTURE ends. Once the
    crack leaves the patch it runs through the homogeneous embedding and then
    into the outer boundary, where the surfing solution breaks down - J there
    says nothing about the microstructure and would badly contaminate Gc_eff.
    (Measured on the synthetic test case: including the embedding tail gave
    Gc_eff = 9.74 with 35 % scatter, clipping to the patch gives 11.23 with
    11 % - and the clipped value is insensitive to frac_lo.)

    Lower end: a fixed fraction of that span, to skip the initial transient.

    The window is defined on the CRACK TIP POSITION, not on time, so runs with
    different time stepping stay comparable. It does not require the run to be
    finished - a partly traversed patch still has a usable plateau over the
    part it did cross, and how much that was is reported separately as
    `patch_fraction_traversed`.
    """
    x = d['x_crack_tip']
    x0 = meta.get('crack_tip_x0_um', float(np.nanmin(x)))
    # start of the microstructure: with a run-in the tip begins at negative x,
    # and only the part inside the patch may enter Gc_eff
    x0 = max(x0, meta.get('patch_x0_um') if meta.get('patch_x0_um') is not None else x0)
    x_reached = float(np.nanmax(x))
    patch_end = (meta.get('microstructure') or {}).get('patch_Lx_um')
    hi = min(x_reached, patch_end) if patch_end else x_reached
    span = max(hi - x0, 1e-9)
    lo = x0 + frac_lo * span
    sel = (x >= lo) & (x <= hi) & np.isfinite(d['Jx'])
    return sel, (lo, hi)


def runin_window(d, meta, frac_lo=0.5):
    """Indices where the crack tip is still in the HOMOGENEOUS embedding,
    before it reaches the patch (crack tip x < 0).

    If the run was started with a `tip_setback`, this stretch is a reference
    measured in the very same run, on the same mesh, with the same time
    stepping. The ratio Gc_eff(patch) / Gc_eff(run-in) is therefore a much
    cleaner statement of what the microstructure does than comparing against
    the nominal Gc: the systematic bias of the discretisation (the homogeneous
    verification run overshoots Gc by ~7 %) cancels out of the ratio.
    """
    x = d['x_crack_tip']
    x0 = meta.get('crack_tip_x0_um', float(np.nanmin(x)))
    patch_x0 = meta.get('patch_x0_um')
    if patch_x0 is None or x0 >= patch_x0:
        return np.zeros_like(x, dtype=bool), (np.nan, np.nan)
    lo = x0 + frac_lo * (patch_x0 - x0)      # skip the initial transient
    sel = (x >= lo) & (x < patch_x0) & np.isfinite(d['Jx'])
    return sel, (lo, patch_x0)


def patch_coverage(d, meta):
    """Fraction of the microstructure patch the crack tip actually crossed.
    None for runs without a patch (homogeneous verification)."""
    Lx = (meta.get('microstructure') or {}).get('patch_Lx_um')
    if not Lx:
        return None
    x0 = meta.get('crack_tip_x0_um', 0.0)
    return float((np.nanmax(d['x_crack_tip']) - x0) / max(Lx - x0, 1e-9))


def zone_windows(meta):
    """FE x-ranges [um] of the three regions inside the patch, from the ROI
    metadata. Only meaningful for the un-rotated (longitudinal) case, where
    the zones vary along the crack direction."""
    ms = meta.get('microstructure', {})
    if ms.get('rotated_ccw90'):
        return {}
    roi = meta.get('roi_um')
    zones = meta.get('zones_um')
    if not roi or not zones:
        return {}
    x0 = roi[0]
    return {'17-4PH': (0.0, zones[0] - x0),
            'transition': (zones[0] - x0, zones[1] - x0),
            '316L': (zones[1] - x0, roi[2] - x0)}


def stats(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                std=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()))


def K_MPa_sqrt_m(Gc_GPa_um, E_prime_GPa):
    return float(np.sqrt(E_prime_GPa * 1e9 * Gc_GPa_um * 1e3) / 1e6)


def evaluate(tag, outdir):
    gpath = os.path.join(outdir, f'run_fracture_simulation_{tag}_graphs.txt')
    mpath = os.path.join(outdir, f'run_meta_{tag}.json')
    if not os.path.isfile(gpath):
        raise SystemExit(f'missing {gpath} - run run_fracture_simulation.py --tag {tag} first')
    meta = json.load(open(mpath)) if os.path.isfile(mpath) else {}
    d = read_graphs(gpath)

    grew, growth = crack_grew(d, meta)
    sel, (lo, hi) = steady_window(d, meta)
    Gc = meta.get('Gc_GPa_um', float('nan'))
    Ep = meta.get('E_prime_GPa', float('nan'))
    if not grew:
        sel = np.zeros_like(sel)        # nothing steady to average

    res = {
        'tag': tag,
        # A run without --micro carries no microstructure at all; calling it
        # "in Aufbaurichtung" would let it into the anisotropy ratio below,
        # where it has no business being.
        'direction': ('homogen (Verifikation)' if not meta.get('micro') else
                      'quer zur Aufbaurichtung' if meta.get('rotate_ccw90')
                      else 'in Aufbaurichtung'),
        'sfun': meta.get('sfun'),
        'Gc_input_GPa_um': Gc,
        'E_prime_GPa': Ep,
        'J_reference_GPa_um': meta.get('J_reference_GPa_um'),
        'n_steps': int(len(d['t'])),
        'crack_tip_range_um': [float(np.nanmin(d['x_crack_tip'])),
                               float(np.nanmax(d['x_crack_tip']))],
        'steady_window_x_um': [float(lo), float(hi)],
        'crack_growth_um': growth,
        'crack_propagated': bool(grew),
        'patch_fraction_traversed': patch_coverage(d, meta),
        'Jx_steady': stats(d['Jx'][sel]),
        'Jy_steady': stats(d['Jy'][sel]),
    }
    _fin = d['Jx'][np.isfinite(d['Jx'])]
    res['Jx_last'] = float(_fin[-1]) if _fin.size else float('nan')

    # A Gc_eff exists only if the steady window actually contains samples.
    # There are two ways it can be empty, and BOTH must produce a warning -
    # otherwise the summary printer looks for a 'warning' key that is not there.
    if res['Jx_steady'].get('n'):
        gce = res['Jx_steady']['mean']
        res['Gc_eff_GPa_um'] = gce
        res['Gc_eff_over_Gc'] = gce / Gc if Gc else None
        res['K_eff_MPa_sqrt_m'] = K_MPa_sqrt_m(gce, Ep) if np.isfinite(Ep) else None
        res['Gc_eff_scatter_rel'] = (res['Jx_steady']['std'] / gce) if gce else None
        # in-run homogeneous reference from the run-in stretch
        rsel, (rlo, rhi) = runin_window(d, meta)
        if rsel.sum() >= 3:
            rs = stats(d['Jx'][rsel])
            res['runin'] = dict(rs, x_range_um=[float(rlo), float(rhi)],
                                Gc_eff_GPa_um=rs['mean'],
                                scatter_rel=rs['std'] / rs['mean'] if rs['mean'] else None)
            res['Gc_eff_over_runin'] = gce / rs['mean'] if rs['mean'] else None
        cov = res['patch_fraction_traversed']
        if cov is not None and cov < 0.5:
            res['warning'] = (
                f'Der Riss hat erst {100*cov:.0f} % des Mikrostruktur-Patches '
                f'durchlaufen ({growth:.0f} um). Gc_eff mittelt damit ueber zu '
                'wenige Koerner und ist NICHT repraesentativ - als Zwischenstand '
                'lesen, nicht berichten. Mit --max_steps 0 zu Ende rechnen.')
    else:
        if not grew:
            res['warning'] = (
                f'Der Riss ist nicht gewachsen (Spitze bewegte sich '
                f'{growth:.3g} um, Schwelle 2*epsilon). Es gibt kein '
                'quasistationaeres Plateau, also auch kein Gc_eff. Typische '
                'Ursachen: zu wenige Zeitschritte (--max_steps), K_scale zu '
                'klein, oder der Lauf steckt noch im Anfangstransienten '
                '(J faellt dann monoton Richtung J_reference_GPa_um).')
        else:
            res['warning'] = (
                f'Die Rissspitze ist zwar {growth:.3g} um gewandert, aber nicht '
                f'bis in das quasistationaere Fenster [{lo:.1f}, {hi:.1f}] um '
                f'(Spitze erreichte {np.nanmax(d["x_crack_tip"]):.1f} um). '
                'Mehr Zeitschritte rechnen (--max_steps 0) oder das Fenster '
                'anpassen.')

    zw = zone_windows(meta)
    if zw:
        res['per_zone'] = {}
        for nm, (a, b) in zw.items():
            m = sel & (d['x_crack_tip'] >= a) & (d['x_crack_tip'] < b)
            st = stats(d['Jx'][m])
            if st.get('n', 0) >= 3:
                st['Gc_eff_GPa_um'] = st['mean']
                st['K_eff_MPa_sqrt_m'] = K_MPa_sqrt_m(st['mean'], Ep) if np.isfinite(Ep) else None
                st['x_range_um'] = [a, b]
                res['per_zone'][nm] = st

    with open(os.path.join(outdir, f'gc_eff_{tag}.json'), 'w') as fh:
        json.dump(res, fh, indent=2)
    return res, d, sel


def make_figure(runs, outdir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available - figure skipped')
        return None

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=False)
    for tag, (res, d, sel) in runs.items():
        lbl = f"{tag} ({res['direction']}, s={res['sfun']})"
        ax[0].plot(d['x_crack_tip'], d['Jx'], lw=0.9, label=lbl)
        if res.get('Gc_eff_GPa_um'):
            ax[0].axhline(res['Gc_eff_GPa_um'], ls='--', lw=0.9, alpha=0.7)
        ax[1].plot(d['t'], d['x_crack_tip'], lw=1.0, label=lbl)
    gcs = {r['Gc_input_GPa_um'] for r, _, _ in runs.values()}
    for g in gcs:
        if np.isfinite(g):
            ax[0].axhline(g, color='k', ls=':', lw=1.2)
    ax[0].set_xlabel('Rissspitze x [um]')
    ax[0].set_ylabel('J_x [GPa*um = kJ/m^2]')
    ax[0].set_title('Fernfeld-J waehrend des Risswachstums '
                    '(gepunktet: eingegebenes Gc, gestrichelt: Gc_eff)')
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel('Rissspitze x [um]')
    ax[1].set_title('Risswachstum')
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, 'fig_J_vs_crack_tip.png')
    fig.savefig(p, dpi=150)
    print(f'geschrieben: {p}')
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('tags', nargs='*', help='run tags to evaluate')
    ap.add_argument('--glob', default=None,
                    help='pattern for graphs files instead of explicit tags')
    ap.add_argument('--outdir', default=here)
    args = ap.parse_args()

    tags = list(args.tags)
    if args.glob:
        for p in sorted(globmod.glob(os.path.join(args.outdir, args.glob))):
            base = os.path.basename(p)
            t = base.replace('run_fracture_simulation_', '').replace('_graphs.txt', '')
            if t not in tags:
                tags.append(t)
    if not tags:
        raise SystemExit('nothing to evaluate: give tags or --glob')

    runs, summary = {}, {}
    for t in tags:
        res, d, sel = evaluate(t, args.outdir)
        runs[t] = (res, d, sel)
        summary[t] = {k: res.get(k) for k in
                      ('direction', 'sfun', 'Gc_input_GPa_um', 'Gc_eff_GPa_um',
                       'Gc_eff_over_Gc', 'K_eff_MPa_sqrt_m',
                       'Gc_eff_scatter_rel', 'crack_propagated',
                       'crack_growth_um', 'warning', 'Gc_eff_over_runin',
                       'runin')}
        summary[t]['patch_fraction_traversed'] = res.get('patch_fraction_traversed')
        summary[t]['per_zone'] = {k: {kk: v[kk] for kk in
                                      ('Gc_eff_GPa_um', 'K_eff_MPa_sqrt_m', 'n')}
                                  for k, v in res.get('per_zone', {}).items()}

    # anisotropy ratio between the two crack directions where both exist
    long_tags = [t for t, s in summary.items() if s['direction'] == 'in Aufbaurichtung']
    tran_tags = [t for t, s in summary.items() if s['direction'] == 'quer zur Aufbaurichtung']
    if long_tags and tran_tags:
        summary['_anisotropy'] = {}
        for lt in long_tags:
            for tt in tran_tags:
                gl, gt = summary[lt].get('Gc_eff_GPa_um'), summary[tt].get('Gc_eff_GPa_um')
                if not (gl and gt):
                    continue
                key = f'{tt}/{lt}'
                summary['_anisotropy'][key] = gt / gl
                # only compare runs whose Gc_eff is representative
                cl = summary[lt].get('patch_fraction_traversed') or 0.0
                ct = summary[tt].get('patch_fraction_traversed') or 0.0
                if min(cl, ct) < 0.5:
                    summary['_anisotropy'].setdefault('_caveat', {})[key] = (
                        f'nicht belastbar: Patch nur zu {100*cl:.0f} % '
                        f'({lt}) bzw. {100*ct:.0f} % ({tt}) durchlaufen')

    with open(os.path.join(args.outdir, 'gc_eff_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    print('\n=== Gc_eff (Mittel von J_x im quasistationaeren Fenster) ===')
    hdr = (f'{"tag":<16}{"Richtung":<24}{"s(x)":<11}{"Gc_eff":>9}{"/Gc":>7}'
           f'{"/Vorlauf":>9}{"K_eff":>8}{"Streu":>7}{"Patch":>7}')
    print(hdr); print('-' * len(hdr))
    stalled = []
    for t in tags:
        s = summary[t]
        if s.get('Gc_eff_GPa_um') is None:
            note = ('-- kein Risswachstum --' if not s.get('crack_propagated')
                    else '-- Fenster nicht erreicht --')
            print(f'{t:<16}{s["direction"]:<24}{str(s["sfun"]):<11}{note:>40}')
            stalled.append(t)
            continue
        cov = s.get('patch_fraction_traversed')
        cov_s = '   n/a' if cov is None else f'{100*cov:5.0f}%'
        mark = '  (!)' if (cov is not None and cov < 0.5) else ''
        rr = s.get('Gc_eff_over_runin')
        rr_s = '      -' if rr is None else f'{rr:9.3f}'
        print(f'{t:<16}{s["direction"]:<24}{str(s["sfun"]):<11}'
              f'{s["Gc_eff_GPa_um"]:>9.4f}'
              f'{s["Gc_eff_over_Gc"] or float("nan"):>7.3f}'
              f'{rr_s}'
              f'{s["K_eff_MPa_sqrt_m"] or float("nan"):>8.1f}'
              f'{(s["Gc_eff_scatter_rel"] or float("nan")):>7.3f}'
              f'{cov_s:>7}{mark}')
        if s.get('warning'):
            stalled.append(t)
    print('  "/Vorlauf" = Gc_eff geteilt durch das J-Plateau im homogenen '
          'Vorlauf DESSELBEN Laufs\n  (systematischer Diskretisierungsfehler '
          'kuerzt sich heraus - das ist die belastbarste Zahl).')
    print('  Gc_eff in GPa*um = kJ/m^2, K_eff in MPa*sqrt(m); "Patch" = von der '
          'Rissspitze durchlaufener\n  Anteil der Mikrostruktur, (!) = < 50 % und '
          'damit nicht repraesentativ. > 100 % heisst, der Riss\n  hat den Patch '
          'verlassen; in Gc_eff geht nur der Abschnitt INNERHALB des Patches ein.')
    for t in dict.fromkeys(stalled):          # keep order, drop duplicates
        r = runs[t][0]
        jl = r.get('Jx_last', float('nan'))
        jr = r.get('J_reference_GPa_um')
        print(f'\n  {t}: {r.get("warning", "kein Gc_eff ermittelbar.")}\n'
              f'    letztes J_x = {jl:.4g}, J_reference = '
              f'{jr if jr is None else round(jr, 4)}, '
              f'Rissfortschritt = {r.get("crack_growth_um", float("nan")):.3g} um, '
              f'Rissspitze {r["crack_tip_range_um"][0]:.1f} ... '
              f'{r["crack_tip_range_um"][1]:.1f} um')
    if summary.get('_anisotropy'):
        caveats = summary['_anisotropy'].get('_caveat', {})
        print('\nAnisotropieverhaeltnis Gc_eff(quer)/Gc_eff(laengs):')
        for k, v in summary['_anisotropy'].items():
            if k == '_caveat':
                continue
            print(f'  {k}: {v:.4f}' + (f'   [{caveats[k]}]' if k in caveats else ''))

    make_figure(runs, args.outdir)
    print(f'geschrieben: {os.path.join(args.outdir, "gc_eff_summary.json")}')


if __name__ == '__main__':
    main()
