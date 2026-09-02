#!/usr/bin/env python3
"""
Inverse Identifikation der Uebergangszonen-Faktoren gegen die DIC-Zielwerte —
Modellhierarchie (Projekt 072):

    A:  keine Parameter        (aK = aCp = aC44 = 1, Baseline = 070 s=1)
    B:  1 Parameter  s         (aK = aCp = aC44 = s, exakt 070 s=const)
    C:  3 Parameter  aK, aCp, aC44

Zielgroesse ist der Zonenmodul der Uebergangszone E_trans (DIC-Grenzflaechen-
wert, Default 232.4 GPa). Da nur region 1 skaliert wird, bewegen sich die
Moduln der Monobereiche praktisch nicht — sie sind KEINE nutzbaren Ziele.
Modell C hat damit 3 Parameter fuer im Wesentlichen EIN Ziel und ist
absichtlich unterbestimmt: das Skript berechnet die Sensitivitaeten
g_i = dE_trans/da_i an a=1, loest das linearisierte Problem MIT MINIMALER
NORM ||a-1|| und weist die verbleibende 2-parametrige Loesungsfamilie
explizit aus (jede Wahl mit g.(a-1) = Delta trifft das Ziel in 1. Ordnung).
Eine eindeutige Aufloesung braucht zusaetzliche Information (Chemie/EDS,
Phasenanteile, weitere Lastrichtungen) — siehe README.

Engines:
    --engine numpy    scipy-Q1-Referenzsolver, laeuft auf dem Host.
                      NUR EXPLORATION — Ergebnispolitik: berichtete Zahlen
                      kommen aus dolfinx.
    --engine dolfinx  ruft solve_plane_stress_eigen.py als Subprozess auf
                      (im alex-dolfinx-Container ausfuehren; --np fuer MPI).

Beispiele (im Container):
    python3 fit_eigen.py --model B --engine dolfinx --np 4
    python3 fit_eigen.py --model C --engine dolfinx --np 4
Host-Exploration:
    python3 fit_eigen.py --model C --engine numpy

Schreibt fit_<model>_<engine>.json (Verlauf, Sensitivitaeten, Endwerte) und
gibt den dolfinx-Bestaetigungsaufruf aus, wenn mit numpy gefittet wurde.
"""
import argparse, json, os, subprocess, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MICRO = os.path.join('..', '070-waam-n1-transition-2d', 'micro_roi.npz')


def eval_numpy(micro, factors, strain):
    import materials_eigen_2d as M2
    import reference_solver_numpy as RS
    cfg = M2.load_config(None, HERE)
    afuns = M2.make_factor_funs(*(repr(float(f)) for f in factors))
    out, _ = RS.solve(micro, cfg, afuns, eps0=strain, verbose=False)
    return out


def eval_dolfinx(micro, factors, strain, tag, nproc):
    cmd = [sys.executable, os.path.join(HERE, 'solve_plane_stress_eigen.py'),
           '--micro', micro, '--tag', tag, '--strain', str(strain),
           '--aK', repr(float(factors[0])), '--aCp', repr(float(factors[1])),
           '--aC44', repr(float(factors[2]))]
    if nproc > 1:
        cmd = ['mpirun', '-np', str(nproc)] + cmd
    subprocess.run(cmd, check=True, cwd=HERE,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    with open(os.path.join(HERE, f'E_{tag}.json')) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=['A', 'B', 'C'], required=True)
    ap.add_argument('--engine', choices=['numpy', 'dolfinx'], default='numpy')
    ap.add_argument('--micro', default=DEFAULT_MICRO)
    ap.add_argument('--target-trans', type=float, default=232.4,
                    help='DIC-Zielwert E_Uebergang [GPa] (Kennwerte_Zugversuche)')
    ap.add_argument('--strain', type=float, default=1e-3)
    ap.add_argument('--fd-step', type=float, default=0.05,
                    help='Schrittweite der Differenzenquotienten (Modell C)')
    ap.add_argument('--tol', type=float, default=0.05,
                    help='Zielgenauigkeit |E_trans - Ziel| [GPa]')
    ap.add_argument('--np', dest='nproc', type=int, default=1,
                    help='MPI-Ranks fuer --engine dolfinx')
    args = ap.parse_args()

    micro = (args.micro if os.path.isabs(args.micro)
             else os.path.join(HERE, args.micro))
    tag = f'fit{args.model}'
    history = []

    def run(factors, label):
        factors = np.asarray(factors, float)
        if args.engine == 'numpy':
            out = eval_numpy(micro, factors, args.strain)
        else:
            out = eval_dolfinx(micro, factors, args.strain, tag, args.nproc)
        rec = {'label': label, 'aK': factors[0], 'aCp': factors[1],
               'aC44': factors[2],
               'E_trans_GPa': out['zone_transition']['E_local_GPa'],
               'E_apparent_GPa': out['E_apparent_GPa'],
               'E_17-4PH_GPa': out['zone_17-4PH']['E_local_GPa'],
               'E_316L_GPa': out['zone_316L']['E_local_GPa']}
        history.append(rec)
        print(f"  [{label}] a=({factors[0]:.4f},{factors[1]:.4f},{factors[2]:.4f})"
              f"  E_trans={rec['E_trans_GPa']:.2f} GPa"
              f"  E_app={rec['E_apparent_GPa']:.2f} GPa")
        return rec

    print(f'Modell {args.model}, Engine {args.engine}, Ziel '
          f'E_trans = {args.target_trans} GPa, micro = {os.path.basename(micro)}')
    base = run([1.0, 1.0, 1.0], 'baseline (A)')
    result = {'model': args.model, 'engine': args.engine, 'micro': micro,
              'target_trans_GPa': args.target_trans, 'baseline': base}

    if args.model == 'B':
        # sekantenverfahren auf s; Startschaetzer aus der Fast-Linearitaet
        s0, e0 = 1.0, base['E_trans_GPa']
        s1 = args.target_trans / e0
        rec = run([s1] * 3, 'B secant 1')
        e1 = rec['E_trans_GPa']
        it = 1
        while abs(e1 - args.target_trans) > args.tol and it < 6:
            s0, e0, s1 = s1, e1, s1 + (args.target_trans - e1) * (s1 - s0) / (e1 - e0)
            rec = run([s1] * 3, f'B secant {it + 1}')
            e1 = rec['E_trans_GPa']
            it += 1
        result['fitted'] = {'s': s1, **rec, 'n_solves': len(history)}

    elif args.model == 'C':
        # Sensitivitaeten an a = 1 (Differenzenquotienten)
        h = args.fd_step
        g = np.empty(3)
        for k, nm in enumerate(('aK', 'aCp', 'aC44')):
            fa = np.ones(3); fa[k] += h
            g[k] = (run(fa, f'FD {nm}')['E_trans_GPa'] - base['E_trans_GPa']) / h
        delta = args.target_trans - base['E_trans_GPa']
        result['sensitivities_GPa_per_unit'] = {'dE/daK': g[0], 'dE/daCp': g[1],
                                                'dE/daC44': g[2]}
        # Minimalnorm-Loesung des linearisierten Problems g.(a-1) = delta
        a = 1.0 + g * delta / float(g @ g)
        rec = run(a, 'C min-norm')
        it = 0
        while abs(rec['E_trans_GPa'] - args.target_trans) > args.tol and it < 4:
            delta2 = args.target_trans - rec['E_trans_GPa']
            a = a + g * delta2 / float(g @ g)      # Gauss-Newton mit fixem g
            rec = run(a, f'C GN {it + 1}')
            it += 1
        result['fitted'] = {'aK': a[0], 'aCp': a[1], 'aC44': a[2], **rec,
                            'n_solves': len(history)}
        result['non_uniqueness'] = (
            'UNTERBESTIMMT: 3 Parameter, 1 informatives Ziel. Jede Wahl mit '
            f'g.(a-1) = {delta:.2f} GPa trifft das Ziel in 1. Ordnung '
            '(2-parametrige Familie). Die angegebene Loesung minimiert '
            '||a-1||. Eindeutigkeit erfordert zusaetzliche Daten '
            '(EDS-Komposition, Phasenanteile, weitere Lastrichtungen).')

    result['history'] = history
    fout = os.path.join(HERE, f'fit_{args.model}_{args.engine}.json')
    with open(fout, 'w') as fh:
        json.dump(result, fh, indent=2)
    print(f'\nwrote {os.path.basename(fout)}  ({len(history)} Solverlaeufe)')
    if 'fitted' in result:
        f = result['fitted']
        print(f"Endwerte: E_trans = {f['E_trans_GPa']:.2f} GPa "
              f"(Ziel {args.target_trans}), E_app = {f['E_apparent_GPa']:.2f} GPa")
    if args.engine == 'numpy' and args.model in ('B', 'C'):
        f = result['fitted']
        aK, aCp, aC44 = f.get('aK', f.get('s')), f.get('aCp', f.get('s')), f.get('aC44', f.get('s'))
        print('\nERGEBNISPOLITIK: berichtete Zahlen kommen aus dolfinx. '
              'Bestaetigen mit:\n'
              f"  python3 solve_plane_stress_eigen.py --micro {args.micro} "
              f"--tag fit{args.model}_final --aK {aK:.6f} --aCp {aCp:.6f} "
              f"--aC44 {aC44:.6f}")


if __name__ == '__main__':
    main()
