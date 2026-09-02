# 072 — WAAM N=1 Übergangsbereich: eigen-skalierte Steifigkeit (K, C', C44)

Ziel: die in 070 eingeführte skalare Stellschraube s(x) der Übergangszone
verallgemeinern. Statt den ganzen kubischen Tensor mit **einem** Faktor zu
skalieren, werden die **drei irreduziblen Anteile** des kubischen
Einkristalltensors separat und ortsabhängig skaliert — eine freiere, aber
weiterhin kubische Steifigkeit, die sich später ggf. über die chemische
Zusammensetzung (EDS) oder Phasenanteile rechtfertigen lässt.

Grundlage ist unverändert die **explizit aus dem EBSD-Scan rekonstruierte
Mikrostruktur** aus 070 (`../070-waam-n1-transition-2d/micro_roi.npz` wird
direkt wiederverwendet, kein neues Preprocessing).

## Parametrisierung

Der kubische Tensor hat drei irreduzible Anteile (Eigenräume):

    K   = (C11 + 2·C12)/3     Kompressionsmodul       (Eigenwert 3K)
    C'  = (C11 − C12)/2       tetragonaler Schub      (Eigenwert 2C')
    C44                       trigonaler Schub        (Eigenwert 2C44)

In der Übergangszone (region 1, wie in 070) wird zellweise

    C_Zelle = P( R(g_Korn) · [ aK(x)·Ch + aCp(x)·Ct + aC44(x)·Cs ] )

gebildet: `Ch/Ct/Cs` sind die drei Anteile der Konstanten aus `config.json`
(Kristallsystem des Korns), `R` die Bond-Rotation mit den Bunge-Winkeln des
Korns, `P` die exakte Kondensation auf den ebenen Spannungszustand. Die
Faktoren sind beliebige numpy-Ausdrücke in x [µm] (`--aK/--aCp/--aC44`),
Skalierung **vor** der Rotation im Kristallsystem.

Eigenschaften (alle in `selftest.py` numerisch belegt):

* `aK = aCp = aC44 = s(x)` reproduziert **exakt** 070 (`--sfun` setzt alle
  drei; Rotation und Kondensation sind homogen in C).
* Positive Faktoren ⇒ positiv definit, ohne weitere Bedingungen — der Grund,
  K/C'/C44 statt C11/C12/C44 zu skalieren.
* Der Zener-Faktor skaliert mit A → (aC44/aCp)·A: aCp und aC44 verändern die
  **Anisotropie**, nicht nur die Größe. aC44 lässt E⟨100⟩ unverändert und
  bewegt E⟨111⟩ — die drei Faktoren wirken physikalisch unterscheidbar.

## Dateien / Ablauf

| Datei | Zweck |
|---|---|
| `plane_stress_crystal.py` | Kristallmathe aus 070 (unverändert) + Eigenzerlegung (`cubic_eigen_parts`, `cubic_scaled`, …) |
| `materials_eigen_2d.py` | **gemeinsame** Materialzuordnung beider Solver: region → Material → Konstanten, Rotation, Kondensation, Faktoren aK/aCp/aC44 nur in region 1. `python3 materials_eigen_2d.py` zeigt die Konstanten inkl. K/C'/C44-Split |
| `config.json` | Einkristallkonstanten (Platzhalter, identisch 069/070) — einzige Editierstelle |
| `solve_plane_stress_eigen.py` | dolfinx v0.7.3, standalone — **einzige Ergebnisquelle** |
| `reference_solver_numpy.py` | numpy/scipy-Prüfrechnung, schreibt nur nach `verification/` |
| `selftest.py` | 29 Unit-Tests (070er-Tests + Eigenzerlegung), `python3 selftest.py` → ALL PASS |
| `fit_eigen.py` | inverse Identifikation, Modellhierarchie A/B/C (s.u.) |
| `run_fem.sh` | Standardfälle |

```bash
# im dolfinx-Container:
bash run_fem.sh                      # base, s133, k160, cp160, c44_160
NP=4 bash run_fem.sh
python3 fit_eigen.py --model B --engine dolfinx --np 4
python3 fit_eigen.py --model C --engine dolfinx --np 4

# auf dem Host (nur Verifikation/Exploration):
python3 selftest.py
python3 reference_solver_numpy.py --micro ../070-waam-n1-transition-2d/micro_roi.npz --tag base
python3 fit_eigen.py --model C --engine numpy
```

Standardfälle in `run_fem.sh`: `base` (alles 1 = 070 `roi`), `s133`
(alle 1.33 = 070 `roi_s133`, Äquivalenzcheck), `k160`/`cp160`/`c44_160`
(je **ein** Faktor 1.60 — zeigt, welcher Anteil wieviel am Zonenmodul bewegt).

## Modellhierarchie und Identifizierbarkeit (fit_eigen.py)

| Modell | Parameter | Bedeutung |
|---|---|---|
| A | 0 | reine Mikrostruktur (= 070 s=1): E_Übergang = 175.2 GPa, Messwert 232.4 GPa wird NICHT erreicht |
| B | 1 (s) | konstante Überhöhung des ganzen Tensors (= 070 s≈1.33) |
| C | 3 (aK, aCp, aC44) | Anteile separat — **absichtlich unterbestimmt** |

Da nur region 1 skaliert wird, bewegen sich die Moduln der Monobereiche
praktisch nicht; informativ ist im Wesentlichen **ein** Ziel (E_Übergang,
DIC 232.4 GPa). Modell C hat dafür 3 Parameter: `fit_eigen.py` berechnet
daher die Sensitivitäten dE/da_i an a=1, löst das linearisierte Problem mit
**minimaler Norm** ‖a−1‖ und weist die verbleibende 2-parametrige
Lösungsfamilie explizit aus. Das ist der zentrale methodische Punkt des
Projekts: die freiere Steifigkeit **kann** den Messwert treffen, ist aber aus
dem einen Zugversuch **nicht eindeutig** identifizierbar — Eindeutigkeit
erfordert zusätzliche Information (EDS-Kompositionsprofil über dC_ij/dc,
Phasenanteile, weitere Lastrichtungen/Schubversuche). Genau diese Kopplung an
Messgrößen ist der vorgesehene nächste Schritt (s. Offene Punkte).

## Verifikation (Stand Anlage des Projekts, 2026-09-02)

* `selftest.py`: 29/29 PASS (inkl. Roundtrip K/C'/C44, Summe der Anteile,
  Voigt-Matrix-Eigenwerte, 070-Äquivalenz über die volle Kette,
  Zener-Skalierung, Positivdefinitheit, build_cell_tensors-Äquivalenz).
* numpy-Prüfrechnung `--tag base` (alle Faktoren 1) auf `micro_roi.npz`:
  E_apparent = 182.865200236 GPa — identisch mit dem in 070 dokumentierten,
  dolfinx-verifizierten Wert (182.865200238, rel. Abw. 1e-11); Zonenmoduln
  187.037 / 175.194 / 185.963 GPa ebenfalls identisch.
* numpy-Prüfrechnung `--sfun 1.33`: E_Übergang = 232.37 GPa, E_app =
  198.09 GPa — trifft die 070-Werte (232.4 / 198.1).
* Die dolfinx-Läufe selbst (`run_fem.sh`, `compare_check` analog 070) stehen
  noch aus — **erst danach dürfen Zahlen berichtet werden.**

## Offene Punkte

* dolfinx-Läufe im Container (`bash run_fem.sh`, Fits mit `--engine dolfinx`)
  + Kreuzvergleich gegen `verification/`.
* Kovariaten-Kopplung: Faktoren nicht frei, sondern aus kornweisen Messgrößen
  (EDS-Komposition × Literatur-Sensitivitäten dC_ij/dc, Phasenanteil,
  ggf. Bandkontrast) — dann ist die Ortsabhängigkeit eine Vorhersage mit
  wenigen globalen Parametern statt eines Fits.
* Höhenstudie (Bänder/full aus 070/study_rois.py) mit dem besten Modell.
* Wie in 069/070: Einkristallkonstanten sind Literatur-Platzhalter; ebener
  Spannungszustand; die 2D-Extrusion kompensiert ggf. fehlende 3D-Kornform —
  frei gefittete Faktoren absorbieren auch diese Modellfehler (im Bericht als
  Confounder benennen).
