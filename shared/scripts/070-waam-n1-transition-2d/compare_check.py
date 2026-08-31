#!/usr/bin/env python3
"""Cross-check: compare dolfinx result (E_<tag>.json) against the numpy
reference (E_<tag>_ref.json). Usage: python3 compare_check.py <tag>"""
import json, os, sys

import numpy as np

tag = sys.argv[1] if len(sys.argv) > 1 else 'roi'
here = os.path.dirname(os.path.abspath(__file__))
fa = f'E_{tag}.json'                       # dolfinx = the result
fb = os.path.join('verification', f'E_{tag}_ref.json')   # numpy = check only
try:
    A = json.load(open(os.path.join(here, fa)))
except FileNotFoundError:
    sys.exit(f'{fa} fehlt - erst solve_plane_stress.py im dolfinx-Container laufen lassen.')
try:
    B = json.load(open(os.path.join(here, fb)))
except FileNotFoundError:
    sys.exit(f'{fb} fehlt - erst reference_solver_numpy.py laufen lassen '
             f'(schreibt nach verification/).')

print(f'Verifikation: dolfinx-ERGEBNIS ({fa}) vs. numpy-Pruefrechnung ({fb}):\n')
ok = True
def row(name, a, b, tol):
    global ok
    d = abs(a - b) / max(abs(b), 1e-12)
    good = d < tol
    ok = ok and good
    print(f'  {name:28s} {a:12.4f} {b:12.4f}  rel.Abw. {d:.2e}  {"OK" if good else "!! ABWEICHUNG"}')

row('E_apparent [GPa]', A['E_apparent_GPa'], B['E_apparent_GPa'], 1e-4)
row('nu_xy', A['nu_xy_apparent'], B['nu_xy_apparent'], 1e-3)
for z in ['17-4PH', 'transition', '316L']:
    ka = A.get(f'zone_{z}'); kb = B.get(f'zone_{z}')
    if ka and kb:
        row(f'E_lokal {z} [GPa]', ka['E_local_GPa'], kb['E_local_GPa'], 1e-4)
pa, pb = A.get('E_profile'), B.get('E_profile')
if pa and pb and len(pa['E_GPa']) == len(pb['E_GPa']):
    ea, eb = np.array(pa['E_GPa']), np.array(pb['E_GPa'])
    rms = float(np.sqrt(np.mean((ea - eb)**2)) / np.mean(eb))
    good = rms < 1e-4
    ok = ok and good
    print(f'  {"E(x)-Profil rel. RMS":28s} {rms:.2e}  {"OK" if good else "!! ABWEICHUNG"}')
print('\n==> ' + ('BESTANDEN: beide Solver stimmen ueberein.' if ok
                  else 'NICHT BESTANDEN: Abweichungen pruefen (Netz? sfun? config?).'))
sys.exit(0 if ok else 1)
