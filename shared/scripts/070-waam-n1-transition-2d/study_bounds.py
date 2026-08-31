#!/usr/bin/env python3
"""
Auswertung der Höhenstudie OHNE FE-Rechnung.

Für jeden Fall aus `study_rois.py` werden aus den kornweisen, rotierten und
plane-stress-kondensierten Tensoren berechnet:

  * je Zone die Voigt- und die Reuss-Schranke (solverunabhängig),
  * je Zone das arithmetische Mittel der kornweisen Richtungsmoduln — laut
    Anhang C des Berichts ein Schätzer, der die FE-Lösung im Mittel auf < 0,1
    GPa und örtlich auf wenige Prozent trifft,
  * der daraus folgende erforderliche Vorfaktor s = 232,4 / E_Übergang,
  * Streifenprofile (≈100 µm) derselben drei Größen,
  * Korn- und Phasenstatistik.

Alles hier ist reine Materialzuordnung (materials_2d) — kein Solver. Die
FE-Bestätigung läuft getrennt über `run_fem.sh` (dolfinx).

Aufruf:  python3 study_bounds.py [--cases roi band1 …]
"""
import argparse, json, os

import numpy as np

import materials_2d as M2

HERE = os.path.dirname(os.path.abspath(__file__))
E_EXP_TRANS = 232.4                     # gemessener Grenzflächenwert [GPa]
ZONE_NAME = {0: '17-4PH', 1: 'Übergang', 2: '316L'}


def voigt_reuss(C, mask):
    """(Reuss, Voigt) E_x [GPa] der Zellen in `mask` (flächengleiche Zellen)."""
    Csel = C[mask]
    Cv = Csel.mean(axis=0)
    Sr = np.linalg.inv(Csel).mean(axis=0)
    return float(1.0 / Sr[0, 0]), float(1.0 / np.linalg.inv(Cv)[0, 0])


def strips(nx, step, target=100.0):
    """Streifenindizes (Liste von (i0, i1)) mit ≈ `target` µm Breite."""
    k = max(1, int(round(nx * step / target)))
    edges = np.linspace(0, nx, k + 1).astype(int)
    return list(zip(edges[:-1], edges[1:]))


def case_stats(tag, cfg):
    d = np.load(os.path.join(HERE, f'micro_{tag}.npz'))
    euler, phase, gid, zone = d['euler_deg'], d['phase'], d['grain_id'], d['zone']
    xc, yc = d['x_um'], d['y_um']
    meta = json.loads(str(d['meta']))
    step = meta['step_um']
    ny, nx = phase.shape
    C, Ex, _, info = M2.build_cell_tensors(euler, phase, gid, zone, xc, cfg,
                                           verbose=False)
    Cf = C.reshape(-1, 3, 3)
    zf, Ef, gf, pf = zone.ravel(), Ex.ravel(), gid.ravel(), phase.ravel()

    out = dict(tag=tag, label=meta.get('case_label', tag), roi_um=meta['roi_um'],
               zones_um=meta['zones'], zone_source=meta.get('zone_source', ''),
               n_cells=int(ny * nx), ny=int(ny), nx=int(nx), step_um=step,
               n_grains=int(len(np.unique(gid))),
               E_x_cell_range=[float(Ex.min()), float(Ex.max())], zones={})

    for z in (0, 1, 2):
        m = zf == z
        if not m.any():
            continue
        reuss, voigt = voigt_reuss(Cf, m)
        ids, cnt = np.unique(gf[m], return_counts=True)
        out['zones'][ZONE_NAME[z]] = dict(
            area_frac=float(m.mean()),
            n_grains=int(len(ids)),
            largest_grain_area_frac=float(cnt.max() / m.sum()),
            f_bcc=float((pf[m] == 2).mean()),
            E_reuss=reuss, E_voigt=voigt,
            E_arith=float(Ef[m].mean()),
            E_harm=float(1.0 / (1.0 / Ef[m]).mean()))

    tr = out['zones'].get('Übergang')
    if tr:
        out['s_required_from_arith'] = float(E_EXP_TRANS / tr['E_arith'])
        out['exp_over_voigt'] = float(E_EXP_TRANS / tr['E_voigt'])

    # Kontrollrechnung: dieselben Zellen, aber mit den Zonengrenzen des
    # Referenzbandes (666,6 / 1494,5 µm) — trennt den Einfluss der
    # Zonendefinition vom Einfluss der Mikrostruktur.
    zref = np.where(xc < 666.6, 0, np.where(xc < 1494.5, 1, 2))
    zreff = np.broadcast_to(zref, (ny, nx)).ravel()
    out['zones_fixedref'] = {}
    for z in (0, 1, 2):
        m = zreff == z
        if not m.any():
            continue
        reuss, voigt = voigt_reuss(Cf, m)
        out['zones_fixedref'][ZONE_NAME[z]] = dict(
            area_frac=float(m.mean()), E_reuss=reuss, E_voigt=voigt,
            E_arith=float(Ef[m].mean()))

    prof = []
    for i0, i1 in strips(nx, step):
        m = np.zeros((ny, nx), bool); m[:, i0:i1] = True
        mf = m.ravel()
        reuss, voigt = voigt_reuss(Cf, mf)
        prof.append(dict(x_um=float(0.5 * (xc[i0] + xc[i1 - 1])),
                         E_arith=float(Ef[mf].mean()),
                         E_reuss=reuss, E_voigt=voigt,
                         zone_major=int(np.bincount(zf[mf]).argmax())))
    out['profile'] = prof
    out['n_distinct_grain_tensors'] = info['n_distinct_grain_tensors']
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'study_stats.json'))
    args = ap.parse_args()

    cases = args.cases
    if cases is None:
        cj = os.path.join(HERE, 'study_cases.json')
        cases = ([c['tag'] for c in json.load(open(cj))] if os.path.isfile(cj)
                 else ['roi'])
    cfg = M2.load_config(here=HERE)

    res = []
    for tag in cases:
        print(f'{tag} …', flush=True)
        s = case_stats(tag, cfg)
        res.append(s)
        tr = s['zones'].get('Übergang', {})
        print(f"   Übergang: Reuss {tr.get('E_reuss',0):.1f}  "
              f"arith {tr.get('E_arith',0):.1f}  Voigt {tr.get('E_voigt',0):.1f} GPa"
              f"   |  Messwert/Voigt = {s.get('exp_over_voigt',0):.2f}"
              f"   s_erf = {s.get('s_required_from_arith',0):.2f}")
    json.dump(dict(E_exp_transition_GPa=E_EXP_TRANS, cases=res),
              open(args.out, 'w'), indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
