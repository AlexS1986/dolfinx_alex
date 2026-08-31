# AGENTS.md — 071 WAAM N=1 2D-Phasenfeld-Bruch, J-Integral, Gc_eff

`README.md` in diesem Ordner zuerst lesen.

## Pfade (damit dieser Ordner wiederfindbar ist)

* **dolfinx-Projektwurzel:**
  `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex`
  (`docker-compose.yml`, `Dockerfile` → `dolfinx/dolfinx:v0.7.3`;
  `shared/` ist im Container `/home`, `PYTHONPATH=/home/utils`)
* **dieses Projekt:** `.../dolfinx_alex/shared/scripts/071-waam-n1-fracture-2d`
  → im Container `/home/scripts/071-waam-n1-fracture-2d`
* Vorgänger/Nachbarn: `.../shared/scripts/070-waam-n1-transition-2d` (Zugfall,
  Steifigkeit), `.../shared/scripts/069-waam-polycrystal-anisotropy`,
  `.../shared/scripts/067_plasticity_inclusions` (Vorbild Phasenfeld + J)
* Bibliothek: `.../dolfinx_alex/shared/utils/alex`
* EBSD-Rohdaten (außerhalb des Container-Mounts!):
  `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/Meshing/Neper/data/04_anisotropy_waam/data_c04/uebergangsbereich/`
* Neper-Pipeline und deren Doku:
  `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/Meshing/Neper/data/04_anisotropy_waam/neper_pipeline/{AGENTS.md,README.md,documentation.txt}`

## Abgrenzung (hart)

071 ist ein **eigener Ansatz**. Nichts aus 071 geht in den 070-Bericht
(`070-waam-n1-transition-2d/report/`) oder in dessen Zahlen ein, und 071
verändert weder 069/070 noch `alex/`. Wenn 071 einen Bericht bekommt, dann
einen eigenen unter `071-.../report/`. Aus 070 wird nur benutzt:
`preprocess_ebsd_to_grid.py` (unverändert aufgerufen) und die
Einkristallkonstanten (als **Kopie** in `config.json`).

## Regeln

* **Materialzuordnung nur an einer Stelle:** `config.json` +
  `materials_fracture_2d.py`. `crystal2d.py` ist eine bewusste Kopie der
  Rotations-/Kondensationsmathematik aus `070/plane_stress_crystal.py`;
  `selftest_material.py` prüft die Gleichheit, solange 070 erreichbar ist.
  Nie eine der beiden Seiten allein ändern.
* **Gc ist per Studiendesign konstant** (auch über Materialien und Regionen).
  Es liegt trotzdem als DG0-Feld vor. Wer das ändert, muss es im Bericht und
  hier explizit machen — sonst ist die Kernaussage („Richtungsabhängigkeit
  kommt allein aus der Elastizität“) hinfällig.
* **Voigt-Konvention:** 3D `[xx,yy,zz,yz,xz,xy]`, 2D `[xx,yy,xy]`,
  **Ingenieur-Gleitungen** überall. `psi_el = ½ eps_v·C·eps_v` ist damit
  identisch zu `½ σ:ε` (in `selftest_material.py` geprüft).
* **Frames:** npz-Eulerwinkel sind TSL-MAP-Frame (y nach unten), FE ist y-up →
  immer über `crystal2d.FLIP_X180`. Der Querfall bekommt zusätzlich `ROT_Z90`,
  und das Gitter wird mit `rotate_grid_ccw90` mitgedreht. Nicht durch
  Vorzeichenwechsel an den Winkeln „reparieren“.
* **`micro_trans.npz` immer mit `--rotate_ccw90`** rechnen und das Netz mit
  `--rotated` erzeugen. Sonst passen Patchmaße und Orientierungen nicht.
* **Änderungen an der Kristallmathematik** müssen `python3 selftest_material.py`
  auf ALL PASS halten; Änderungen an der Phasenfeldklasse
  `python3 selftest_phasefield.py` (im Container). Tests erweitern, nicht
  aufweichen. Der isotrope Grenzfall gegen
  `alex.phasefield.StaticPhaseFieldProblem2D` ist der wichtigste Test — er
  darf nie schlechter als 1e-12 relativ werden.
* **dolfinx-API:** über `dolfinx_compat.py` gehen (0.7.3 im Container vs. 0.8+
  auf Clustern). Keine direkten `dlfx.fem.FunctionSpace`/`functionspace`-Aufrufe
  in neuen Skripten.
* **Vor jedem längeren Lauf** `bash run_local_test.sh` (Minuten, braucht keine
  EBSD-Daten). Vor belastbaren Aussagen zusätzlich eine Netz-/ε-Studie.
* **Einheiten GPa/µm.** 1 GPa·µm = 1 kJ/m², **1 GPa·√µm = 1 MPa·√m (exakt)**.
  Der K-Faktor ist wirklich 1 (31,62 gilt für GPa·√mm) und steht nur an einer
  Stelle: `materials_fracture_2d.GPA_SQRT_UM_TO_MPA_SQRT_M`. Nie mischen, nie
  eine zweite Umrechnung irgendwo hineinschreiben. Gegenprobe bei jeder
  Änderung: `K = sqrt(Gc·E)` in Modelleinheiten muss zahlengleich mit
  `sqrt(E[Pa]·Gc[J/m²])/1e6` sein. Jeder Lauf schreibt `run_meta_<tag>.json`
  mit allen Umrechnungen.
* **`run_meta` immer über `json_safe`/`write_run_meta` schreiben.** Bounding box
  und `dlfx.fem.Constant.value` sind numpy und sonst nicht serialisierbar.
* **Startzustand des gemischten Feldes über `set_intact_state`** (s=1, u=0) für
  `w` UND `wm1`. `w.sub(1).x.array` ist in dolfinx der GANZE gemischte Vektor,
  kein s-Ausschnitt — direktes Zuweisen darauf setzt auch u. Wird `w` nicht
  initialisiert, startet Newton bei s≡0 (überall gebrochen): erst viele
  gescheiterte Schritte, dann ein langer Heilungstransient mit **fallendem**
  `A_surf` und monoton fallendem `J` gegen `J_reference`, ohne Risswachstum.
* **Anfangsriss mit analytischem Profil** `s = 1-exp(-d/2eps)` (`initial_phasefield`),
  nicht als scharfer Kerb. Der exakte 1D-Minimierer des Oberflächenterms; das
  Querintegral ist 1 pro Risslänge, also `A = Risslänge` ab Schritt 1. Jeder
  Lauf druckt `A/L` als Selbstkontrolle — Sollwert ≈ 1. Ein scharfer Kerb gibt
  `A/L = eps/h` und verschenkt die ersten ~40 Schritte an Relaxation.
  `--no_init_profile` nur zum Vergleichen.
* **P1 für `u` UND `s`**, fest verdrahtet, kein Schalter. Nicht "aufbohren":
  - `s` MUSS P1 sein. `alex.phasefield.irreversibility_bc` sucht die s-DOF über
    `locate_entities(domain, 0, ...)` (nur Ecken) und
    `alex.postprocessing.crack_bounding_box_2D` indiziert eine Maske der Länge
    #s-DOF gegen `domain.geometry.x` (#Ecken). Mit P2-`s` heilt der Riss an den
    Kantenmittelknoten aus und die Rissverfolgung ist falsch — beides ohne
    Fehlermeldung. Wer P2-`s` will, muss vorher beide alex-Funktionen ersetzen.
  - `u` ist P1 per Entscheidung (Nutzer, 2026-08-21): um einen regularisierten
    Riss ist die Verschiebung nicht singulär, P2 kostet ~3x DOF und bringt
    wenig. Genauigkeit kommt über eps/h, nicht über den Polynomgrad.
* **Startwerte:** `u` aus dem analytischen Surfing-K-Feld
  (`initial_displacement`), `s` aus dem Rissprofil (`initial_phasefield`).
  Ohne den u-Startwert springt der erste Newton-Schritt von 0 auf die volle
  Randbedingung; daran ist der erste Produktionslauf gestorben (`dt` auf 9e-15).
  `--no_init_u` / `--no_init_profile` nur zum Vergleichen.
* **Anfangsriss beginnt VOR dem Patch** (`--tip_setback`, Default
  `min(10*eps, 0.4*linker Rand)`), vollständig im homogenen Material. Zweck:
  Prozesszone bei t=0 außerhalb der Mikrostruktur, und der homogene Vorlauf
  liefert ein J-Plateau als laufinterne Referenz. Der Lauf druckt Vorlauflänge
  in Vielfachen von epsilon und warnt bei < 2*eps oder wenn die Spitze im Patch
  liegt. Der alte Default `0.25*Lx_dom` hat 18 % des Patches übersprungen.
* **`Gc_eff/Vorlauf` ist die belastbarste Kennzahl**, nicht `Gc_eff/Gc`: der
  systematische Diskretisierungsfehler (homogener Lauf überschätzt Gc um ~7,5 %)
  kürzt sich im Verhältnis heraus. Auch das Anisotropieverhältnis quer/längs aus
  den vorlaufnormierten Werten bilden. Die Streuung im Vorlauf muss klein sein
  (~0.002); ist sie es nicht, war der Vorlauf zu kurz.
* **K-Feld-Zentrum startet am linken Gebietsrand** (`K_START_X = x_min`,
  `--k_start_x`), NICHT auf der Rissspitze — wie in 067 (`crack_start = [0,...]`
  bei einem Gebiet ab x=0). Damit gehört das aufgebrachte Feld anfangs zu einem
  kürzeren Riss als dem realen, die Spitze ist unterbelastet und bewegt sich
  erst, wenn das Feld sie eingeholt hat. Das ist die Anlauframpe des
  Surfing-Verfahrens; zentriert man auf der Spitze, liegt ab t=0 der volle
  Antrieb an.
* **Rissstartposition NICHT ändern.** `crack_tip_x0 = x_min + a0` mit
  `a0 = -x_min - tip_setback`, `tip_setback = min(10*eps, 0.4*(-x_min))`. Sie
  hängt damit nur von Netz und epsilon ab. Wer an `Tend`, `crack_width`,
  `separation_frac` oder `v_crack` dreht, darf sie nicht mitverschieben —
  sonst sind Läufe untereinander nicht mehr vergleichbar.
* **`dt` hängt NICHT an `Tend`.** `dt_max = 0.33*epsilon/v_crack` (das K-Feld
  wandert ein Drittel der Phasenfeldlänge je Schritt), `dt_start = dt_max/200`.
  Eine frühere Version koppelte beides an `Tend`, sodass ein größeres Zeitbudget
  stillschweigend die Zeitschrittweite mitveränderte (0.032 -> 0.060).
* **Zeitschritt:** `dt` wächst im alex-Stepper nur bei `iters < min_iters`.
  alex-Default 4 ist zu streng (Läufe hängen bei 4–5 Iterationen fest); 071
  nutzt `--min_iters 5`. `dt_start` klein anfangen (1e-5·Tend), `dt_max`
  1e-3·Tend. Nicht an diesen Defaults drehen, ohne `newton_iters` und `dt` in
  der Graphs-Datei anzuschauen.
* **Graphs-Spalten nur ANHÄNGEN**, nie umsortieren — `evaluate_gc_eff.COLUMNS`
  und `GRAPH_LABELS` in `run_fracture_simulation.py` müssen zusammenpassen.
* **Ergebnisquelle:** ausschließlich `run_fracture_simulation.py` (dolfinx).
  `make_test_micro.py synth` erzeugt **synthetische** Daten (`meta.synthetic`);
  daraus darf keine berichtete Zahl stammen.
* **Fernfeld-J ist Absicht:** im heterogenen Körper ist der Eshelby-Tensor
  nicht divergenzfrei, das Konturintegral über den Außenrand enthält die
  Materialkräfte der Mikrostruktur. Das ist die gesuchte effektive
  Energiefreisetzungsrate — nicht als „Fehler in der Wegunabhängigkeit“
  wegdiskutieren oder auf eine Spitzenkontur umbauen, ohne das zu benennen.
