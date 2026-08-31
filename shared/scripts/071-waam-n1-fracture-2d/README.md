# 071 — WAAM N=1: 2D-Phasenfeld-Risssimulation mit J-Integral und effektiver Bruchzähigkeit

**Neuer, eigenständiger Ansatz.** Dieses Projekt ist bewusst von 069
(Polykristall-Anisotropie) und 070 (Steifigkeit im Übergangsbereich, Zugfall)
**abgegrenzt**: eigene Konfiguration, eigene Skripte, eigene Doku, eigener
Bericht. Es geht **nicht** in den 070-Bericht
(`070-waam-n1-transition-2d/report/`) ein und darf ihn nicht verändern.
Übernommen wird aus 070 nur zweierlei: die EBSD-Vorverarbeitung
(`preprocess_ebsd_to_grid.py`, unverändert aufgerufen) und die
Einkristallkonstanten (als Kopie in der eigenen `config.json`).

---

## Wo alles liegt (absolute Pfade)

| Was | Pfad |
|---|---|
| **dolfinx-Projektwurzel** (Docker/Apptainer, `docker-compose.yml`) | `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex` |
| Container-Mount `shared` → `/home` | `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex/shared` |
| **dieses Projekt** | `.../dolfinx_alex/shared/scripts/071-waam-n1-fracture-2d` (im Container `/home/scripts/071-waam-n1-fracture-2d`) |
| Vorgänger Zugfall (Steifigkeit) | `.../dolfinx_alex/shared/scripts/070-waam-n1-transition-2d` |
| Vorbild Phasenfeld + J-Integral | `.../dolfinx_alex/shared/scripts/067_plasticity_inclusions` |
| gemeinsame Bibliothek `alex` | `.../dolfinx_alex/shared/utils/alex` (im Container `PYTHONPATH=/home/utils`) |
| **EBSD-Rohdaten** (Übergangsbereich) | `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/Meshing/Neper/data/04_anisotropy_waam/data_c04/uebergangsbereich/WAAM_N=1_A12D_Uebergangsbereich.{txt,bmp}` |
| Neper-Pipeline (Kornstatistik, Doku) | `/Users/alexanderschluter/Work/Hypo/Hypo/Simulation/Meshing/Neper/data/04_anisotropy_waam/neper_pipeline` |

Die EBSD-Rohdaten liegen **außerhalb** des Container-Mounts — die
Vorverarbeitung läuft deshalb auf dem Host, alles andere im Container.

---

## Ziel

Der Riss wächst durch die **explizit aus dem EBSD-Scan rekonstruierte
Mikrostruktur** des N=1-Übergangsbereichs. Jedes Korn trägt seinen **eigenen
rotierten Einkristall-Steifigkeitstensor** — dieselbe lokale Steifigkeit wie im
Zuglastfall von 070, inklusive der drei Überhöhungsvarianten s(x) im
Übergangsbereich. **Gc ist überall konstant.** Damit ist die einzige Quelle
einer richtungsabhängigen effektiven Bruchzähigkeit die elastische
Heterogenität/Anisotropie der Kornstruktur.

Zwei Fälle:

| Fall | Mikrostruktur | Rissrichtung |
|---|---|---|
| `long_*` | `micro_long.npz` — ROI von 070 (17-4PH → Übergang → 316L entlang x) | entlang +x = **in Aufbaurichtung**, quer durch die Übergangszone |
| `trans_*` | `micro_trans.npz` — hoher Streifen **innerhalb** der Übergangszone, mit `--rotate_ccw90` um 90° gedreht | entlang +x im FE-Frame, physikalisch **quer zur Aufbaurichtung**, bleibt in der Übergangszone |

Aufbaurichtung ist im EBSD-Scan −x (siehe 070). Die Drehung um 90° dreht Gitter
**und** Orientierungen mit (`crystal2d.ROT_Z90`), sodass der Surfing-Rand und die
ganze Auswertung in beiden Fällen entlang +x laufen können.

---

## Modell

### Elastizität — kornweiser Tensor

Wie im Zugfall von 070, Einheiten GPa/µm:

    C_Zelle(x) = s(x_map) · P( R(g_Korn) · C_kubisch[Material][Kristallsystem] )

`R` Bond-Rotation mit den Bunge-Winkeln **dieses Korns** (Map→FE-Flip
`FLIP_X180`, im Querfall zusätzlich `ROT_Z90`), `P` exakte statische
Kondensation auf den ebenen Spannungszustand (Voigt `[xx, yy, xy]`,
Ingenieur-Gleitungen), `s(x)` der Vorfaktor **nur** in der Übergangszone.

Außerhalb des Mikrostruktur-Patches: **homogener isotroper Stahl** mit
**angenommenen** Konstanten (`config.json → embedding`, Default E = 200 GPa,
ν = 0,3). Das ist der bewusste Unterschied zu 067 — hier wird **nicht**
homogenisiert, die Einbettung ist nur eine elastische Umgebung, damit der
Surfing-Rand weit genug von der Rissspitze entfernt liegt.

### Bruch — die neue Klasse

`phasefield_anisotropic.StaticPhaseFieldProblem2D_anisotropic`

    eps_v(u) = [u_x,x , u_y,y , u_x,y + u_y,x]
    sigma    = g(s, eta) · C(x) · eps_v(u)
    psi_el   = ½ · g(s, eta) · eps_v · C(x) · eps_v   ( = ½ σ:ε )
    Pi       = ∫ ( psi_el + Gc·( (1-s)²/(4ε) + ε |∇s|² ) ) dx
    Res      = iMob·(s-s_{n-1})/Δt·δs + δPi/δs + δPi/δu

Alle Klassen in `alex.phasefield` parametrisieren das Elastizitätsgesetz über
(λ, μ) und sind damit zellweise **isotrop**. Diese Klasse ersetzt das Paar
(λ, μ) durch **eine** DG0-Funktion `C` (3×3 pro Zelle) — die Signatur von
`prep_newton` ist ansonsten identisch, sodass das Treiberskript austauschbar
bleibt. Bewusste Entscheidungen:

* **Kein Zug-Druck-Split.** Die volle elastische Energie wird degradiert, wie in
  `alex.phasefield.StaticPhaseFieldProblem2D` und in 067. Für ein allgemeines
  anisotropes C ist ein Split ohne Zusatzannahmen nicht wohldefiniert.
* **Gc als Feld, aber konstant.** Gc liegt als DG0-Feld vor (damit später eine
  Gc-Heterogenität zuschaltbar ist), ist in dieser Studie aber überall gleich.
* **Irreversibilität** wie in 067: Dirichlet-Bedingung
  `alex.phasefield.irreversibility_bc` plus viskoser Ratenterm.

Die Klasse liegt **in diesem Ordner**, nicht in `alex/phasefield.py`. So bleibt
die gemeinsame Bibliothek unberührt; wenn sich der Ansatz bewährt, kann sie
später nach `alex` wandern.

### Last — Surfing-Randbedingung (wie 067)

Auf dem Außenrand wird das Mode-I-Verschiebungsfeld

    u_i = K/(2μ√(2π)) · √r · (κ − cos θ) · {cos(θ/2), sin(θ/2)}

vorgeschrieben, dessen Zentrum mit `v_crack` entlang +x wandert. κ und E′
richten sich nach dem Zustand (`plane_state`):

| | κ | E′ |
|---|---|---|
| ebener Spannungszustand (Default) | (3−ν)/(1+ν) | E |
| ebener Verzerrungszustand | 3−4ν | E/(1−ν²) |

`K = K_scale · √(Gc·E′)`, sodass `K_scale = 1` genau die Fernfeld-Triebkraft
`J = K²/E′ = Gc` des homogenen Referenzfalls erzeugt. `K_scale` etwas über 1
(Default in `run_cases.sh`: 1,05) treibt den Riss zuverlässig an.

### Auswertung — J und Gc_eff

`J = ∮ Σ·n ds` über den **Außenrand**, mit dem elastischen Eshelby-Tensor
`Σ = psi_el,deg · I − ∇u^T · σ_deg`.

Wichtig und bewusst so: im heterogenen Körper ist `div Σ ≠ 0`, wo C
ortsabhängig ist. Der Fernfeld-J enthält deshalb die Rissspitzen-Triebkraft
**plus** die konfigurationellen Kräfte der Mikrostruktur innerhalb der Kontur —
genau das ist die makroskopisch wirksame Energiefreisetzungsrate. Ihr Plateau
während des quasistationären Wachstums ist

    Gc_eff = ⟨J_x⟩ (stationäres Fenster),    K_eff = √(E′·Gc_eff)

`evaluate_gc_eff.py` bildet das Fenster über die **Rissspitzenposition** (nicht
über die Zeit), sodass Läufe mit unterschiedlicher Zeitschrittweite
vergleichbar bleiben, und zerlegt J(x_Spitze) im Längsfall zusätzlich nach den
drei Zonen.

---

## Einheiten

Durchgehend **GPa und µm**. Daraus folgt:

| Größe | Einheit | Umrechnung |
|---|---|---|
| C, E | GPa | — |
| Gc, J | GPa·µm | **1 GPa·µm = 1 kJ/m²** |
| K | GPa·√µm | **1 GPa·√µm = 1 MPa·√m** (exakt) |

Die K-Umrechnung ist wirklich der Faktor 1:
`1 GPa·√µm = 10⁹ Pa · √(10⁻⁶ m) = 10⁹·10⁻³ Pa·√m = 10⁶ Pa·√m = 1 MPa·√m`.
(31,62 wäre der Faktor für GPa·√mm — nicht hier.) Der Faktor steht als
`materials_fracture_2d.GPA_SQRT_UM_TO_MPA_SQRT_M` an genau einer Stelle.

Default `Gc = 10 GPa·µm = 10 kJ/m²` → `K_eq = √(E·Gc) = 44,72 GPa·√µm
= 44,72 MPa·√m`. Gegenprobe in SI: `√(200·10⁹ Pa · 10·10³ J/m²) = 4,472·10⁷
Pa·√m`. Passt.

**Degradationsfunktion und kritische Spannung.** 071 nutzt die **quadratische**
(AT2-)Form `g(s) = s² + η`. Damit folgt aus Gc = 10 GPa·µm und ε = 12 µm

    σ_c = (9/16)·√(Gc·2μ/(6ε)) = 2,60 GPa

— im richtigen Bereich für die Spaltbruchfestigkeit von Stahl. Die kubische
Form aus 067 (β = 0,1) gäbe bei gleichem Gc und ε `σ_c = (81/50)·√(Gc·2μ/(15ε))
= 4,74 GPa`; um damit auf 2,6 GPa zu kommen, müsste man Gc auf 3,0 GPa·µm
senken und K_eq fiele von 44,7 auf 24,6 MPa·√m. Umschaltbar mit
`--degradation cubic`.

Bekannter Preis von AT2: es gibt **keine elastische Grenze**, `s` sinkt schon
bei kleinster Last unter 1. Bei einem vorhandenen Riss unter Surfing-Last ist
das akzeptiert und üblich, muss aber im Bericht stehen. ε bleibt eine
**numerische Regularisierungslänge**, keine Materiallänge. Jeder Lauf druckt
σ_c mit aus.

---

## Dateien

| Datei | Zweck | läuft wo |
|---|---|---|
| `config.json` | **einzige** Stelle für Konstanten: Einkristall-C, Gc, ε, Einbettung, ROIs | — |
| `crystal2d.py` | Kristallmathematik (Bond-Rotation, Kondensation, `ROT_Z90`, Gitterdrehung) | numpy |
| `materials_fracture_2d.py` | Zuordnung Region→Material→Konstanten, `Microstructure` mit Punktabfrage, Einbettung | numpy |
| `phasefield_anisotropic.py` | **die neue Phasenfeldklasse** | dolfinx |
| `dolfinx_compat.py` | Shim für dolfinx 0.7.3 vs. 0.8+ | dolfinx |
| `mesh_fracture_micro.py` | gmsh-Netz: Patch in Einbettung, verfeinerter Risskorridor, eingebettete Risslinie | gmsh |
| `run_fracture_simulation.py` | Hauptlauf: Surfing-BC, adaptive Zeitschritte, J, XDMF, Graphs | dolfinx |
| `evaluate_gc_eff.py` | J-Plateau → Gc_eff, K_eff, Zonenzerlegung, Vergleichsabbildung | numpy/matplotlib |
| `make_test_micro.py` | kleine Testmikrostruktur (Ausschnitt oder synthetisch) | numpy |
| `selftest_material.py` | 26 numpy-Unit-Tests der Materialkette | numpy |
| `selftest_phasefield.py` | dolfinx-Tests der neuen Klasse, inkl. isotropem Grenzfall gegen `alex` | dolfinx |
| `prepare_micro.sh` | ruft 070s EBSD-Vorverarbeitung für beide ROIs auf | **Host** |
| `run_local_test.sh` | kompletter Rauchtest ohne EBSD-Daten | Container |
| `run_cases.sh` | Produktionsläufe: 2 Richtungen × 3 s-Varianten | Container |

---

## Ablauf

### 0) Container starten (Host)

```bash
cd /Users/alexanderschluter/Work/Hypo/Hypo/Simulation/dolfinx_alex
docker compose up -d
```

### 1) Lokaler Test — ohne EBSD-Daten, wenige Minuten

```bash
docker exec -it alex-dolfinx bash -lc \
  "cd /home/scripts/071-waam-n1-fracture-2d && NP=4 bash run_local_test.sh"
```

Fünf Stufen: numpy-Selbsttest → dolfinx-Selbsttest der Klasse → homogener
Verifikationslauf (J muss gegen K²/E′ laufen) → synthetische Mikrostruktur in
beiden Richtungen → Auswertung. Steuerung: `STAGES="1 2"`, `STEPS=…`, `NP=…`.
Alles landet in `test_out/`.

### 2) Mikrostruktur aus dem EBSD-Scan (Host)

```bash
cd .../dolfinx_alex/shared/scripts/071-waam-n1-fracture-2d
bash prepare_micro.sh          # -> micro_long.npz, micro_trans.npz
```

### 3) Produktionsläufe (Container)

```bash
docker exec -it alex-dolfinx bash -lc \
  "cd /home/scripts/071-waam-n1-fracture-2d && NP=10 bash run_cases.sh"
```

Sechs Fälle: `long_s1 long_s133 long_sgauss trans_s1 trans_s133 trans_sgauss`.
Teilmengen mit `CASES="long_s1 trans_s1"`. Die drei s(x) sind exakt die von
070: `1.0`, `1.33`, `1 + 0.50·exp(−((x−1050)/350)²)` (x in Map-Koordinaten, µm).

### 4) Auswertung

```bash
python3 evaluate_gc_eff.py long_s1 long_s133 long_sgauss \
                           trans_s1 trans_s133 trans_sgauss --outdir results
```

→ `gc_eff_<tag>.json`, `gc_eff_summary.json` (inkl.
`_anisotropy: Gc_eff(quer)/Gc_eff(längs)`), `fig_J_vs_crack_tip.png`.

### Der homogene Vorlauf — laufinterne Referenz

Der Anfangsriss startet um `--tip_setback` (Default `min(10ε, 0,4·linker Rand)`)
**vor** der linken Patchkante, also vollständig im homogenen Stahl. Das hat zwei
Gründe:

1. Die Prozesszone (Breite ~2ε) liegt bei t = 0 komplett außerhalb der
   Mikrostruktur. Mit der Spitze exakt auf x = 0 ragte sie schon halb hinein.
2. Der Riss läuft im homogenen Material in einen stationären Zustand, **bevor**
   die Körner eingreifen. Das J-Plateau dieser Strecke ist eine Referenz aus
   demselben Lauf, auf demselben Netz, mit derselben Zeitschrittweite.

Deshalb meldet die Auswertung neben `Gc_eff/Gc` auch **`Gc_eff/Vorlauf`**. Der
systematische Diskretisierungsfehler — der homogene Verifikationslauf
überschätzt Gc um ~7,5 % — steckt in Zähler und Nenner gleichermaßen und kürzt
sich heraus. **Das ist die belastbarste Zahl des Projekts**, und dasselbe gilt
für das Anisotropieverhältnis: quer/längs sollte man aus den vorlaufnormierten
Werten bilden, nicht aus den rohen.

Die Kennzahlen des Vorlaufs stehen als Block `runin` in `gc_eff_<tag>.json`
(Mittelwert, Streuung, x-Bereich). Seine Streuung muss **klein** sein (~0,002,
wie im homogenen Verifikationslauf) — ist sie es nicht, ist der Vorlauf zu kurz
oder der Riss dort noch nicht stationär.

---

## Ausgabedateien je Lauf

| Datei | Inhalt |
|---|---|
| `run_fracture_simulation_<tag>_graphs.txt` | `t Jx Jy x_crack_tip x_K_field Rx_top Ry_top dW W A_surf dt E_el E_surf E_total` |
| `run_fracture_simulation_<tag>.xdmf/.h5` | ParaView: `u`, `s`, `sigma`, `grain_id`, `E_x_local_GPa`, `region`, `s_prefactor`, `gc_GPa_um`, `in_patch` |
| `run_meta_<tag>.json` | vollständige Provenienz: Konstanten, ROI, K, v_crack, Netz, MPI-Ranks |
| `parameters_<tag>.txt` | dieselben Skalare im `key=value`-Format der Gruppe |

### In ParaView

`alex.postprocessing` legt **pro Feldname eine eigene temporale Collection** im
XDMF an. Ein Feld existiert also nur zu den Zeiten, zu denen es geschrieben
wurde — und ParaView zeigt die Collections als **Blöcke** eines Multiblock-
Datensatzes. Zwei Konsequenzen:

* Steht der Zeitregler auf einem Zeitpunkt, zu dem ein Feld nicht geschrieben
  wurde, fehlt es einfach. 071 schreibt deshalb **alle** Felder zu **jedem**
  Ausgabezeitpunkt inklusive t = 0 (früher lagen die Materialfelder nur bei
  t = 0 und `u`/`s` nur bei t > 0 — dann sieht man bei t = 0 kein `u` und
  kein `s`).
* Fehlt trotzdem etwas: Multi-Block-Inspector öffnen und prüfen, ob der Block
  `u` / `s` / `sigma` überhaupt aktiviert ist.

Coloring-Empfehlung:

| Feld | wozu |
|---|---|
| `E_x_local_GPa` | **der eigentliche Test** der Materialzuordnung — kornweise Steifigkeit, im Patch fleckig, außen glatt der Einbettungswert |
| `grain_id` | Kornstruktur (Colormap „Random”), außerhalb des Patches −1 |
| `in_patch` | Patch gegen Einbettung |
| `region` | −1 außen, 0 = 17-4PH, 1 = Übergang, 2 = 316L |
| `s` | Phasenfeld = Riss (s = 1 intakt, s = 0 gebrochen) |
| `u`, `sigma` | Lösung |

`phase_fcc1_bcc2` hat absichtlich nur zwei Werte (plus 0 außerhalb) — das ist
nur das Kristallsystem, nicht die Kornstruktur.

Schnelle Kontrolle ohne ParaView, direkt aus der HDF5:

```python
import h5py, numpy as np
f = h5py.File('run_fracture_simulation_<tag>.h5')
for k in ['E_x_local_GPa', 'grain_id', 'gc_GPa_um', 'region']:
    a = np.array(f[f'Function/{k}/0']).ravel()
    print(f'{k:16s} {a.min():9.3f} ... {a.max():9.3f}   '
          f'{len(np.unique(np.round(a,4))):6d} verschiedene Werte')
```

Erwartung: `E_x_local_GPa` so viele verschiedene Werte wie Korntensoren + 1
(Einbettung), `gc_GPa_um` genau **einen** Wert, `region` die Werte −1/0/1/2.

---

## Verifikation

`selftest_material.py` (numpy, 26 Checks, läuft überall):
kubisch↔isotrop, Bond-Rotation invertierbar/invarianzerhaltend,
E⟨100⟩/E⟨110⟩/E⟨111⟩ analytisch getroffen, Kondensation gegen E und ν,
`ROT_Z90` (E_x der gedrehten ROI = E_y der ungedrehten, 20 Zufallsorientierungen),
Gitterdrehungs-Indexabbildung (x,y)→(L_y−y, x), Punktabfrage,
Einbettung außerhalb, s(x)-Vorfaktor, Voigt-Energieidentität,
und — falls 070 erreichbar — **bitgenaue Übereinstimmung mit
`070/plane_stress_crystal.py`**.

`selftest_phasefield.py` (dolfinx, im Container):

1. Patch-Test σ = C·ε für allgemeines anisotropes C
2. ψ_el = ½ ε·C·ε = ½ σ:ε
3. **isotroper Grenzfall**: Residuumsvektor, elastische Energie und
   Eshelby-Tensor identisch zu `alex.phasefield.StaticPhaseFieldProblem2D`
   (Toleranz 1e-12 relativ) — das ist der Test, der garantiert, dass die neue
   Klasse nur das Elastizitätsgesetz verallgemeinert, nicht die Physik ändert
4. Degradation skaliert σ und ψ_el exakt mit g(s,η)
5. homogener Körper, homogene Dehnung → Konturintegral von Σ = 0
6. heterogener Körper → Konturintegral ≠ 0 (die Materialkraft wird erfasst)

Zusätzlich Stufe 3 von `run_local_test.sh`: homogener Lauf, `J_x` muss gegen
`K²/E′` laufen (in `run_meta_homog.json` als `J_reference_GPa_um` mitgeschrieben).

---

## Zeitschrittsteuerung — was man wissen muss

Der adaptive Stepper in `alex.solution.solve_with_newton_adaptive_time_stepping`
verhält sich asymmetrisch:

* `dt` wird **halbiert**, wenn Newton nicht konvergiert (`max_iters`, Default 8),
* `dt` wird **verdoppelt**, wenn Newton **weniger** als `min_iters` Iterationen
  braucht — sonst bleibt es, wo es ist.

`alex` setzt `min_iters=4` als Default. Braucht ein Lauf dauerhaft 4–5
Iterationen, wächst `dt` **nie wieder** und der Lauf bleibt bei einem einmal
heruntergehalbelten `dt` stehen. 071 setzt deshalb `--min_iters 5` als Default;
`--max_iters` ist ebenfalls einstellbar.

Zweitens muss `dt` **klein starten**: Defaults sind `dt_start = 1e-5·Tend`
und `dt_max = 1e-3·Tend` (in 067 sind die Verhältnisse ähnlich). Ein großes
`dt_start` wird ohnehin sofort weggehalbelt — nur eben unter Verlust vieler
gescheiterter Newton-Läufe.

### Startzustand des Phasenfelds

Der Anfangsriss wird als Dirichlet-Bedingung `s = 0` auf der Risslinie gesetzt.
Das allein ist ein **scharfer Kerb**: nur eine Knotenreihe hat s = 0, das
regularisierte Profil der Breite ~2ε fehlt noch. Dessen Flächenmaß ist dann

    A/Länge = ε/h     statt     A/Länge = 1

— mit ε = 8 µm und h = 2 µm also viermal zu viel, und der Löser verbringt die
ersten ~40 Schritte damit, das wegzurelaxieren (sichtbar an `A_surf`, das von
860 auf 220 **fällt**, während der Riss nicht wächst).

071 startet deshalb mit dem **analytischen Profil**, optional mit einem
stumpfen Kerb der vollen Breite W (`--crack_width`, Default 24 µm = 2ε):

    s(d) = 0                              für d < W/2   (voll gebrochen, Dirichlet)
    s(d) = 1 − exp( −(d − W/2) / (2ε) )   sonst

mit d = Abstand zum Anfangsriss-Segment

Das ist der exakte 1D-Minimierer des Oberflächenterms: mit v = 1−s liefert die
Euler-Lagrange-Gleichung `v'' = v/(4ε²)`, und das Querintegral über
`(1−s)²/(4ε) + ε|∇s|²` ergibt genau 1 pro Risslänge. Damit hat der Riss von
Schritt 1 an seine Gleichgewichtsbreite, `A` startet gleich bei der Risslänge
und ist danach monoton steigend.

Das Flächenmaß ist dann **A/L = 1 + W/(4ε)** (numerisch gegen die
Querschnittsintegration geprüft, exakt für W = 0…36 µm). Jeder Lauf druckt die
Selbstkontrolle:

```
Startzustand (s: analytisches Profil, u: K-Feld): A = 445.8,
  Risslaenge = 297.2 um, Rissbreite W = 24.0 um -> A/L = 1.50 (Soll 1.50 = 1 + W/(4*eps))
``` Zum Vergleichen gibt es `--no_init_profile`
(scharfer Kerb wie in 067) — dann steht dort `A/L ≈ ε/h`.

### Ansatzgrad: P1 für `u` und `s`

Beide Felder sind **P1**, bewusst und fest verdrahtet.

Für `s` ist es zwingend: zwei Stellen der gemeinsamen `alex`-Infrastruktur sind
eckknotenbasiert und würden mit P2-`s` **stillschweigend falsch** rechnen.

| Stelle | Was sie tut | Warum P2 bricht |
|---|---|---|
| `alex.phasefield.irreversibility_bc` | sucht die s-DOF über `locate_entities(domain, 0, …)` | findet nur Eckknoten; die Kantenmittelknoten würden nie gepinnt, ein gebrochener Riss könnte dort **ausheilen** |
| `alex.postprocessing.crack_bounding_box_2D` | indiziert eine Maske der Länge „Anzahl s-DOF" gegen `domain.geometry.x` (Anzahl Ecken) | die Längen stimmen nur für P1 überein |

Für `u` ist P1 eine Entscheidung, nicht ein Zwang. Die Verschiebung um einen
*regularisierten* Riss ist nicht singulär — die Degradation glättet sie —, das
klassische Argument für einen höheren Ansatzgrad greift hier also nur schwach.
P2-`u` hätte auf Dreiecken etwa viermal so viele Freiheitsgrade, insgesamt grob
das Dreifache, bei direktem LU-Löser eher das Fünffache an Rechenzeit. Was die
Genauigkeit des Phasenfelds tatsächlich bestimmt, ist **ε/h**, also das Netz —
nicht der Polynomgrad.

Wer die verbleibende Abweichung des homogenen Verifikationslaufs
(`Gc_eff/Gc = 1,075`) verkleinern will, dreht deshalb an ε/h, am viskosen Term
oder an der Konturlage, nicht am Ansatzgrad. Für den Richtungsvergleich fällt
sie ohnehin heraus, weil `Gc_eff/Vorlauf` denselben Bias in Zähler und Nenner
hat (siehe oben).

### Weitere Fallstricke

Der Startzustand ist ansonsten **u = 0** — gesetzt über die kollabierten
Sub-Dofmaps `DOFS_U`/`DOFS_S`. Achtung, das ist eine Stolperfalle:
`w.sub(1).x.array` ist in dolfinx **kein** Blick nur auf die s-Freiheitsgrade,
sondern der ganze gemischte Vektor. `w.sub(1).x.array[:] = 1.0` setzt also auch
`u ≡ 1`. Und wird `w` gar nicht initialisiert, startet Newton bei `s ≡ 0`, also
mit überall gebrochenem Material — dann heilt der Löser erst mühsam das ganze
Gebiet zurück, `A_surf` **fällt** dabei, und `J` fällt monoton gegen
`J_reference`, ohne dass der Riss wächst.

### Wenn ein Lauf nicht vorankommt

Diagnose direkt aus der Konsolenausgabe bzw. den Spalten
`newton_iters`, `s_min`, `A_surf`, `dt`, `x_crack_tip`:

| Symptom | Ursache | Abhilfe |
|---|---|---|
| `A_surf` fällt, `x_crack_tip` konstant, `J` fällt monoton gegen `J_reference` | Anfangstransient / Startzustand | `A/L` beim Start prüfen (muss ≈ 1 sein); s=1-Initialisierung prüfen |
| `Gc_eff` gemeldet, aber Spalte „Patch" < 50 % | Riss hat den Patch erst teilweise durchlaufen | `--max_steps 0`; die Zahl bis dahin nicht berichten |
| `dt` einmal gefallen und dann konstant, `newton_iters` = `min_iters` | Stepper kann `dt` nicht mehr anheben | `--min_iters` erhöhen (6, 7) |
| Newton scheitert schon im ersten Schritt mehrfach | `dt_start` zu groß | `--dt_start` senken |
| Riss wächst nie, `J` sauber auf Plateau bei `J_reference` | Triebkraft reicht nicht | `--K_scale` erhöhen (1,1 … 1,3) |
| `s_min` bleibt bei 1 | es entsteht überhaupt kein Schaden | `--K_scale` erhöhen oder `--Gc` senken |

`evaluate_gc_eff.py` meldet „kein Risswachstum" statt eines `nan`, wenn sich
die Rissspitze nicht bewegt hat — ein `Gc_eff` gäbe es dann nicht.

## Grenzen und offene Punkte

* **Einkristallkonstanten sind Literatur-Platzhalter** (identisch zu 069/070) —
  jedes Ergebnis skaliert damit. Bei jeder berichteten Zahl nennen.
* **Gc ist ein freier Parameter**, nicht gemessen. σ_c folgt aus Gc und ε und
  ist mit den Defaults zu hoch (siehe „Einheiten“).
* **Ebener Spannungszustand**, ein einziger Schliff, keine 3D-Constraints.
* **Kein Zug-Druck-Split** — unter überwiegend Mode-I-Zug unkritisch, bei
  Druckanteilen aber eine echte Einschränkung.
* **Korngrenzen sind keine bevorzugten Risspfade** in diesem Modell: Gc ist
  konstant, es gibt keine Kohäsivzone an den Grenzen. Interkristalliner Bruch
  ist damit nicht abbildbar; das Modell zeigt, was allein die elastische
  Heterogenität bewirkt. Ein Gc(Korngrenze) wäre der nächste Schritt und ist im
  Code vorbereitet (Gc liegt bereits als Feld vor).
* **ε ≈ 12 µm liegt in der Größenordnung der feinen 17-4PH-Körner.** Die feine
  Martensit-Struktur ist vom Phasenfeld also nur grob aufgelöst; die groben
  316L-Kolumnarkörner dagegen gut. Netzstudie mit ε = 8 µm und `--e-fine 2`
  vor jeder belastbaren Aussage.
* Die Registrierung EBSD-ROI ↔ Probe (welcher Bereich genau) ist dieselbe
  Annahme wie in 070.
