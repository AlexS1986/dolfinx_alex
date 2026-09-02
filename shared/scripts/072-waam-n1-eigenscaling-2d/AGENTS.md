# AGENTS.md — 072 WAAM N=1 eigen-skalierte Übergangszonen-Steifigkeit

Read `README.md` here first; canonical project docs live in
`.../Meshing/Neper/data/04_anisotropy_waam/neper_pipeline/{AGENTS.md,documentation.txt}`.
072 ist die direkte Verallgemeinerung von 070 (skalares s(x) → drei Faktoren
aK/aCp/aC44 auf die irreduziblen Anteile K, C', C44). 070 bleibt unverändert
— 072 liest nur dessen `micro_*.npz` (Default
`../070-waam-n1-transition-2d/micro_roi.npz`), kopiert nichts zurück.

Material assignment lives in ONE place: `materials_eigen_2d.py` +
`config.json`. Both solvers import it, so they can never drift apart on the
material law. Die Skalierung passiert im KRISTALLSYSTEM vor der Rotation
(`plane_stress_crystal.cubic_eigen_parts`); `aK=aCp=aC44=s` == 070 exakt
(Selftest 17/20). Do not "simplify" the per-column cache in
`build_cell_tensors` into a per-grain cache — inside region 1 the factors
depend on x, so the condensed tensor differs per column.

**Result policy (hard, wie 070):** every reported number, table and figure
comes from `solve_plane_stress_eigen.py` (dolfinx v0.7.3). Der
numpy-Referenzsolver schreibt nach `verification/` und ist reine
Prüfrechnung. `fit_eigen.py --engine numpy` ist Exploration; berichtete
Fit-Ergebnisse mit `--engine dolfinx` rechnen oder den ausgegebenen
Bestätigungsaufruf ausführen.

Rules (inherited from 069/070 + specifics):
- dolfinx **v0.7.3 API** in `solve_plane_stress_eigen.py`
  (ufl.VectorElement/TensorElement + dlfx.fem.FunctionSpace), standalone —
  kein `alex`-Import.
- Voigt 3D [xx,yy,zz,yz,xz,xy], 2D [xx,yy,xy], ENGINEERING shear everywhere.
- Orientation frames: npz Euler angles are TSL MAP frame (y down); FE is y-up
  → always through `plane_stress_crystal.FLIP_X180`. Don't "fix" by negating
  angles.
- Any change to the crystal math must keep `python3 selftest.py` ALL PASS
  (29 Tests); extend the tests when adding functionality.
- Cross-check policy: after changing solver or material module, run the numpy
  reference and dolfinx on the same npz and compare E_*.json (~1e-6 rel.;
  070/compare_check.py lässt sich unverändert auf die JSONs anwenden, wenn
  man ihn hierher kopiert oder die Pfade angibt).
- `config.json`-Konstanten sind Literatur-Platzhalter (identisch 069/070) —
  flag this in any reported number. Die Faktoren aK/aCp/aC44 sind die
  Stellschraube, NICHT die Konstanten; ein eigener "trans"-Eintrag in
  config.json wäre eine konkurrierende Stellschraube — nicht beide zugleich
  drehen.
- **Identifizierbarkeit:** Modell C (3 Parameter) gegen im Wesentlichen 1
  Ziel ist absichtlich unterbestimmt; `fit_eigen.py` berichtet Minimalnorm-
  Lösung + Sensitivitäten + Lösungsfamilie. Diese Nichteindeutigkeit in
  jedem Bericht ausweisen, nie eine der Lösungen als "die" Lösung verkaufen.
- Verifikationsstand bei Anlage (2026-09-02): Selftests ALL PASS; numpy-
  Prüfrechnung base == 070-Referenz (182.8652 GPa, 1e-11) und sfun=1.33 ==
  070 roi_s133 (232.4/198.1 GPa). dolfinx-Läufe stehen aus.

## Maintenance
Update this file (and the canonical docs in the Neper folder: 072-Zeile in
der Projekttabelle) when scripts, their CLI/env, the material model, or the
workflow change.
