# 070 — WAAM N=1 Übergangsbereich: 2D-Plane-Stress-Modell (EBSD-exakt)

Ziel: die im Zugversuch (vertikale Richtung, N=1-Probe 17-4PH → Übergang →
316L) gemessene **Steifigkeitsüberhöhung im Übergangsbereich** in einer
linear-elastischen FE-Rechnung reproduzieren — mit der **explizit aus dem
EBSD-Scan rekonstruierten Mikrostruktur** (kein Neper, kein statistisches
Sampling): reguläres 2D-Netz, Kornzugehörigkeit/Orientierung als Zellfeld.

## Modell

- **Geometrie:** Ausschnitt (ROI) der EBSD-Map
  `WAAM_N=1_A12D_Uebergangsbereich` (4696×3698 µm, 3.371 µm/px,
  IPF-Z-Rendering). Links 17-4PH (BCC-Martensit, fein), rechts 316L
  (FCC, grobe kolumnare Körner), dazwischen Übergangsbereich.
- **Mikrostruktur:** pro Zelle des regulären Gitters Bunge-Winkel + Phase des
  Korns, in dem die Zelle liegt. Die Korn-Zuordnung pro Pixel wird aus dem
  Per-Korn-Export (Zentroid, Ellipse, mittlere Orientierung) + BMP
  rekonstruiert: Kosten = Ellipsen-Mahalanobis (Hinge) + IPF-Farbabstand
  (berechnete IPF-Z-Farbe vs. BMP-Pixel); validiert mit mittlerem
  |ΔRGB| ≈ 17 gegen das Original-BMP.
- **Elastizität:** pro Zelle kubischer Einkristall (FCC → 316L-Konstanten,
  BCC → 17-4PH-Konstanten aus `config.json`, **Platzhalterwerte wie in 069**),
  Bond-rotiert (6×6) mit den Zell-Eulerwinkeln, dann **exakte statische
  Kondensation auf ebenen Spannungszustand** (σ_zz=σ_yz=σ_xz=0):
  `C_red = C_AA − C_AB·C_BB⁻¹·C_BA` (Voigt [xx,yy,xy], Ing.-Gleitungen).
  Beliebige (trikline) rotierte Tensoren sind zulässig.
- **Zonen:** `zone` ∈ {0: 17-4PH, 1: Übergang, 2: 316L} über x-Grenzen
  (`--zones`). In Zone 1 wird C mit dem **skalaren Vorfaktor s(x)**
  multipliziert (`--sfun`, Default `"1.0"`, beliebiger numpy-Ausdruck in
  x [µm]) — Stellschraube für eine ortsabhängige Übergangs-Steifigkeit.
- **Last:** Zug entlang x (= horizontale Map-Achse): u_x=0 links,
  u_x=ε₀·Lx rechts, u_y=0 in einem Eckknoten; oben/unten spannungsfrei.
- **Auswertung:** E_gesamt = ⟨σ_xx⟩/ε₀ sowie je Zone
  E_lokal = ⟨σ_xx⟩_Zone/⟨ε_xx⟩_Zone (Analogon zum lokalen
  DIC-/Extensometer-Modul im Versuch). Felder (u, σ, ε, zone, phase) → XDMF.

## Frames (wichtig)

TSL-Map-Frame: x rechts, **y nach unten**, z in die Ebene (IPF-Z = Map-
Normale, gegen BMP validiert). FE-Frame: x rechts, **y nach oben**, z aus
der Ebene. Umrechnung der Orientierungen über 180°-Drehung um x
(`FLIP_X180` in `plane_stress_crystal.py`); npz-Zeile 0 = oberster Map-Rand.

## Dateien / Ablauf

1. `preprocess_ebsd_to_grid.py` (Host, numpy/scipy/PIL) — EBSD-txt + BMP →
   `micro_<tag>.npz` (+ IPF-/Phasen-Preview-PNG).
   ```bash
   python3 preprocess_ebsd_to_grid.py --txt .../WAAM_N=1_A12D_Uebergangsbereich.txt \
       --bmp .../WAAM_N=1_A12D_Uebergangsbereich.bmp \
       --roi X0 Y0 X1 Y1 --step 3.371 --zones X_PH_ENDE X_TRANS_ENDE --tag roi
   ```
   (ROI/Zonen in µm im Map-Frame, Ursprung oben links, y nach unten.)
2. `solve_plane_stress.py` (dolfinx-Container, v0.7.3, **standalone** — kein
   `alex`-Import) — `TAG=roi SFUN="1.0" bash run_fem.sh`
   → `E_<tag>.json`, `ps_<tag>.xdmf`.
3. `reference_solver_numpy.py` — **nur Verifikation**, kein Ergebnislieferant:
   dasselbe Problem als numpy/scipy-Q1-FEM (läuft ohne dolfinx). Schreibt nach
   `verification/`; `make_figures.py` liest diesen Ordner nie.
4. `selftest.py` — numpy-Unit-Tests der Kristallmathematik (Rotation,
   Kondensation, Isotropie-Grenzfälle). `python3 selftest.py` → ALL PASS.
5. `materials_2d.py` — **gemeinsame** Materialzuordnung beider Solver
   (region → Material → Konstanten je Kristallsystem, Rotation, Kondensation,
   s(x)). Direkt aufrufbar (`python3 materials_2d.py`) zum Anzeigen der
   aktuell verwendeten Konstanten.
6. `compare_check.py <tag>` — Kreuzvergleich dolfinx ↔ numpy-Referenz.
7. `make_figures.py` — Ergebnisfigur (nutzt automatisch die dolfinx-JSONs,
   sobald vorhanden).

## Hoehenstudie: mehrere Baender + volle Probenhoehe (Bericht Kapitel 3)

Der Referenzfall `roi` ist **ein** horizontales Band in der Kartenmitte. Die
Hoehenstudie legt vier weitere Baender gleicher Hoehe darueber/darunter und
zusaetzlich ein Fenster ueber die **volle Kartenhoehe** (horizontal weiterhin
x = 30…2811 µm, das weit rechts liegende homogene 316L bleibt aussen vor).

```bash
# Host (numpy/scipy/PIL): sechs Fenster aus EINER Pixel->Korn-Rekonstruktion
python3 study_rois.py --txt .../WAAM_N=1_A12D_Uebergangsbereich.txt \
                      --bmp .../WAAM_N=1_A12D_Uebergangsbereich.bmp
#   -> micro_band1..4.npz, micro_full.npz, study_cases.json
#      (+ _fullmap_assign.npz = Cache der ganzen Karte, ~90 s beim ersten Mal)

# Host: Schranken/Schaetzer je Fenster, OHNE Solver
python3 study_bounds.py              # -> study_stats.json

# Container: FE-Bestaetigung
NP=8 bash run_study.sh               # alle Faelle, s(x)=1
NP=8 CASES=full bash run_study.sh    # nur das grosse Fenster

# Abbildungen des Kapitels (ergaenzt automatisch die FE-Kurven,
# sobald E_<tag>.json vorliegt)
python3 report/make_study_figs.py --src .../WAAM_N=1_A12D_Uebergangsbereich.bmp
```

* `full` hat 905 025 Zellen = 1,81 Mio. DOF (4x `roi`); CG+GAMG skaliert
  linear, also mit mehreren MPI-Ranks rechnen.
* Zonengrenzen: nur fuer `roi` gibt es die eingezeichnete Markierung. Fuer die
  anderen Fenster bestimmt `study_rois.zone_boundaries` sie aus dem Gefuege
  (Halbwertsstelle des spaltenweisen BCC-Anteils; Beginn des grobkoernigen
  316L bei >80 % Flaechenanteil von d_eq > 500 µm). Am Referenzband kalibriert:
  709/1394 µm gegenueber markierten 667/1495 µm.
* **Ergebnis:** in allen sechs Fenstern liegt der gemessene Grenzflaechenwert
  (232,4 GPa) **oberhalb** der Voigt-Schranke der Uebergangszone (155…211 GPa)
  und damit erst recht oberhalb der FE-Loesung (140…181 GPa). Ueber die volle
  Hoehe faellt zudem der 316L-Wert von 186,0 auf 160,2 GPa (FE) und trifft
  damit den Messwert 162,7 GPa auf 1,6 % — der Einkorn-Artefakt des
  Referenzbandes ist als Stichprobeneffekt belegt.
* FE gerechnet 2026-08-25 (dolfinx v0.7.3, NP=8, 42–45 CG-Iterationen je Fall):
  `E_band1..4.json`, `E_full.json`, `fields_*.npz`, `ps_*.xdmf`. Alle 18
  Zonenmoduln liegen in ihren Schranken; der arithmetische Schaetzer liegt im
  Mittel 2,0 GPa darueber (max. 7,7 GPa).

## Checks

- Patch-Test: homogen-isotrop → E_apparent = E, ν = ν (exakt, verifiziert).
- s(x)=2 homogen → E ×2 (verifiziert).
- ⟨σ_xx⟩ ist in allen Zonen identisch (Serien-Gleichgewicht) — in jedem Lauf
  als Konsistenzcheck ausgegeben.
- Einkristall-Grenzfälle E⟨100⟩/E⟨110⟩/E⟨111⟩ analytisch getroffen.

## Offene Punkte

- ROI-Rechteck + Zonengrenzen aus `WAAM_N=1_A12D_Uebergangsbereich_mit_AR_Bereich.bmp`.
- Experimentelle Zielwerte (E je Bereich) aus `Kennwerte_Zugversuche_WAAM_N=1 (2).pdf`.
- Einkristallkonstanten sind Literatur-Platzhalter (wie 069) — Ergebnisse
  skalieren damit.

## Wo die Kristallsteifigkeiten stehen, und wie sie zugeordnet werden

**Alle Konstanten stehen in `config.json`** in diesem Ordner - das ist die
einzige Stelle zum Editieren. Ausgabe der aktuell verwendeten Werte jederzeit
mit `python3 materials_2d.py`.

Die Zuordnung laeuft in zwei Stufen (Code: `materials_2d.py`):

    region (zone-Tag)  ->  Material     ->  Konstanten, indiziert mit dem
    0                  ->  "17-4PH"         Kristallsystem des KORNS
    1                  ->  "trans"          (fcc/bcc, aus der EBSD-Phase)
    2                  ->  "316L"

    C_Zelle = s(x) * P( R(g_Korn) . C_kubisch[Material][Kristallsystem] )

`R` = Bond-Rotation mit den Bunge-Winkeln **dieses Korns**, `P` = exakte
Kondensation auf den ebenen Spannungszustand, `s(x)` = Vorfaktor, nur in
region 1. Damit hat **jedes Korn seinen eigenen Steifigkeitstensor**: gleiches
Material und gleiches Kristallsystem ergeben trotzdem unterschiedliche
`C`, weil die Orientierung `g` je Korn verschieden ist.

Der Uebergangsbereich hat einen **eigenen Eintrag** `"trans"` in `config.json`
(Default = die Konstanten der Ausgangswerkstoffe, weil die Zusammensetzung
unbekannt ist). Dort koennen eigene C11/C12/C44 eingetragen werden; `s(x)`
bleibt die davon unabhaengige, ortsabhaengige Stellschraube.

Im ROI sind das konkret **7601 verschiedene Koerner**:

| region | Material | Koerner im ROI | Flaechenanteil FCC / BCC |
|---|---|---|---|
| 0 | 17-4PH | 5771 | 0,33 / 0,67 |
| 1 | trans | 1863 | 0,90 / 0,10 |
| 2 | 316L | 18 | 1,00 / 0,00 |

(316L hat im ROI nur 18 Koerner, davon fuellt eines fast die halbe Flaeche -
die grossen kolumnaren Chevron-Koerner. Deshalb wirkt der rechte Bereich in
den Feldbildern fast homogen; er ist es aber nicht per Konstruktion, sondern
weil dort tatsaechlich nur wenige, sehr grosse Koerner liegen.)

## Felder im XDMF (was man in ParaView sieht)

`phase_fcc1_bcc2` hat absichtlich nur zwei Werte - das ist **nur das
Kristallsystem**, nicht die Kornstruktur. Die Kornstruktur und die
kornweise Steifigkeit stehen in diesen Feldern:

| Feld | Bedeutung |
|---|---|
| `grain_id` | Korn-ID je Zelle - **hier ist die Kornstruktur sichtbar** |
| `E_x_local_GPa` | E-Modul in Lastrichtung aus dem *eigenen* Tensor des Korns; variiert im ROI von ~95 bis ~300 GPa |
| `region` | 0 = 17-4PH, 1 = Uebergang, 2 = 316L |
| `phase_fcc1_bcc2` | 1 = fcc, 2 = bcc (nur Buchhaltung) |
| `s_prefactor` | tatsaechlich angewandter Vorfaktor s(x) |
| `u`, `sig_xx/yy/xy`, `eps_xx` | Loesung |

Zum Anschauen in ParaView: `ps_<tag>.xdmf` oeffnen, Coloring auf
`E_x_local_GPa` (oder `grain_id`, Colormap z.B. "Random") stellen, Darstellung
"Surface With Edges" ausschalten. Der numpy-Referenzlauf legt dieselben Felder
mit `--save-fields` in `fields_<tag>_ref.npz` ab
(`Ex_fe`, `grain_fe`, `s_fe`, `zone_fe`, `phase_fe`).

## Bericht

`report/report.md` → `report/WAAM_N1_transition_report.pdf` (13 Seiten, Aufbau
wie der 069-Bericht: Methodik mit allen Formeln und Annahmen, Ergebnisse,
Verifikation, Diskussion, Grenzen, Reproduzierbarkeit).

```bash
python3 report/make_report_figs.py --bmp .../..._mit_AR_Bereich.bmp \
                                   --src .../WAAM_N=1_A12D_Uebergangsbereich.bmp
bash report/build_report.sh          # pandoc + xelatex
```

`make_report_figs.py` erzeugt sieben Abbildungen. Vier haengen nur von
Eingangsdaten und Materialzuordnung ab (ROI-Karte, Rekonstruktionsvergleich,
kornweises E_x, Voigt/Reuss-Schranken), drei kommen aus der FE-Loesung
(sigma/eps-Felder, E(x)-Profil mit Zaehler/Nenner-Zerlegung, DIC-Verlauf).
Die FE-Felder werden aus `fields_roi.npz` gelesen, ersatzweise aus der
XDMF-Begleitdatei `ps_roi.h5` — die schreibt jeder dolfinx-Lauf, unabhaengig
von der Skriptversion. Fehlt beides, werden die FE-Abbildungen mit Hinweis
uebersprungen. Die Kennzahlen des Rechenbeispiels im Bericht landen in
`report/profile_stats.json`.

`fig_averaging.png` (Bericht 3.7) vergleicht die Streifenmittelung der
kornweisen Moduln — arithmetisch, harmonisch/Tensor-Reuss und Tensor-Voigt —
mit der FE-Loesung; die Kennzahlen dazu stehen in `report/averaging_stats.json`.
`fig_grain_Ex_variants.png` zeigt die kornweise Steifigkeit zusaetzlich fuer
beide s-Varianten.

Im E(x)-Diagramm (`fig_E_profile.png`) erscheinen die s-Varianten als
**durchgezogene** Kurven, sobald `E_roi_s133`/`E_roi_gauss` bzw. deren
`ps_*.h5` vorliegen; fehlen sie, werden sie aus der s=1-Loesung abgeleitet und
**gestrichelt** als solche gekennzeichnet (Serienrelation
E_s(x) = s(x)·E_1(x), siehe Bericht 3.6.1). Nach `bash run_fem.sh` also einmal
`python3 report/make_report_figs.py` und `bash report/build_report.sh`
nachziehen — dann stehen dort die echten FE-Profile. **Regel wie in 069: `report.md` und PDF immer gemeinsam
aendern** — nach jeder Textaenderung `build_report.sh` neu laufen lassen. Jede
Abbildung und Tabelle muss im Fliesstext per `\ref{...}` referenziert sein.

## Rechnen (dolfinx-Container) — die einzige Ergebnisquelle

**Alle berichteten Ergebnisse stammen aus `solve_plane_stress.py` (dolfinx
v0.7.3).** Der numpy-Solver ist reines Verifikationswerkzeug, seine Ausgaben
landen in `verification/` und gehen in keine Abbildung und keine Tabelle ein.

```bash
# Host: Container starten
cd .../dolfinx_alex
docker compose up -d                      # bzw. der uebliche Start von alex-dolfinx

# im Container: alle drei Standardfaelle rechnen
docker exec -it alex-dolfinx bash -lc \
  "cd /home/scripts/070-waam-n1-transition-2d && bash run_fem.sh"

# Abbildung aus den dolfinx-Ergebnissen erzeugen
docker exec -it alex-dolfinx bash -lc \
  "cd /home/scripts/070-waam-n1-transition-2d && python3 make_figures.py"
```

`run_fem.sh` rechnet nacheinander:

| tag | s(x) |
|---|---|
| `roi` | `1.0` — reine Mikrostruktur |
| `roi_s133` | `1.33` — konstante Ueberhoehung im Uebergang |
| `roi_gauss` | `1 + 0.50*exp(-((x-1050)/350)^2)` — Gauss-Profil |

Steuerung: `CASES=roi bash run_fem.sh` (nur ein Fall), `NP=4 bash run_fem.sh`
(MPI). Ein eigener Fall direkt:

```bash
python3 solve_plane_stress.py --micro micro_roi.npz --tag meinfall \
        --sfun "1 + 0.3*np.exp(-((x-1050.)/400.)**2)"
```

Jeder Lauf schreibt drei Dateien:

| Datei | Inhalt |
|---|---|
| `E_<tag>.json` | E_apparent, nu, Moduln je Zone, E(x)-Profil, verwendete Konstanten |
| `fields_<tag>.npz` | alle Zellfelder auf dem Mikrostrukturgitter (fuer `make_figures.py`) |
| `ps_<tag>.xdmf/.h5` | ParaView: u, sig, eps, `grain_id`, `E_x_local_GPa`, `region`, `s_prefactor` |

`make_figures.py` liest **ausschliesslich** `E_<tag>.json` und
`fields_<tag>.npz`; fehlt eine Datei, nennt das Skript den noetigen
dolfinx-Aufruf statt still etwas anderes einzusetzen.

### Woher die E-Werte kommen (Panels d/e/f)

**Modellwerte — aus der Loesung des Randwertproblems.** Geloest wird die
lineare Elastizitaet im ebenen Spannungszustand auf dem ROI:

    div sigma = 0,  sigma = C_Zelle(x,y) : eps(u)
    u_x = 0 bei x=0,   u_x = eps0*Lx bei x=Lx  (eps0 = 1e-3)
    u_y = 0 in einem Eckknoten (Starrkoerper)
    oben/unten (y=0, y=Ly): spannungsfrei

Aus der Loesung u wird zellweise sigma_xx und eps_xx berechnet. Daraus:

* **Zonenwert** (Panel d): `E_Zone = <sigma_xx>_Zone / <eps_xx>_Zone`
* **Profil** (Panel e): dasselbe in ~100 µm breiten Streifen entlang x:
  `E(x) = sum(sigma_xx) / sum(eps_xx)` je Streifen

Wichtig: `<sigma_xx>` ist in jedem Streifen praktisch gleich (Serienschaltung,
Gleichgewicht in Lastrichtung) — die gesamte Variation von E(x) steckt in der
**lokalen Dehnung**. Weiche Bereiche dehnen sich staerker, also E kleiner. Das
ist genau die Groesse, die DIC misst: Spannung (aus Kraft/Querschnitt, entlang
der Probe konstant) geteilt durch die oertlich gemessene Dehnung. eps0=1e-3 ist
beliebig, E haengt linear-elastisch nicht davon ab.

**Experimentelle Werte — zwei verschiedene DIC-Auswertungen:**

| | Quelle | Was gemessen wurde |
|---|---|---|
| Panel d + blaue Linien in e | `Kennwerte_Zugversuche_WAAM_N=1 (2).pdf`, Tabelle | E je **Auswertefenster** (316L / Grenzflaeche / 17-4PH), Flaechenmittel ueber Fenster von mehreren mm Breite, Proben MHo1026–1029 |
| Panel f | `Verlauf_Elastizitaetsmodul_lokal_N=1.xlsx` | lokales E an **14 Punkten** entlang der ganzen Messstrecke, Probe MHo1030, bis 1,5 % Dehnung |

**Panel (e) ist damit KEIN Ausschnitt aus Panel (f).** Es sind zwei
unabhaengige Auswertungen, und die x-Achsen sind verschiedene Groessen
(mm im EBSD-ROI vs. Punktindex ueber ~14 mm).

Zur Lesart von Panel (e): die kraeftige blaue Linie ist der DIC-Zonenmittel-
wert, gezeichnet ueber den x-Bereich der zugehoerigen Modellzone; das blasse
Band darum ist die Standardabweichung. Beides teilt sich einen
Legendeneintrag; die gestrichelten Vertikalen sind die Zonengrenzen.

**Wichtige Einschraenkung dieses Vergleichs:** das experimentelle
Auswertefenster "Grenzflaeche" ist mehrere mm breit, die Modell-Uebergangszone
(gruenes Rechteck im EBSD-Scan) nur **0,83 mm**. Der gemessene Wert 232,4 GPa
mittelt also ueber deutlich mehr Material als die Modellzone enthaelt — die
Gegenueberstellung in (d)/(e) ist insofern naeherungsweise. Der 14-Punkt-
Verlauf in (f) zeigt allerdings, dass die Ueberhoehung tatsaechlich lokal ist
(Spitze 237 GPa an einem einzelnen Punkt), was die Zuordnung stuetzt. Die
Registrierung EBSD-ROI <-> DIC-Punktindex ist nicht bekannt; das graue Fenster
in (f) ist eine Annahme (Peak = Grenzflaeche, Innenraster 1 mm).

### Optionale Verifikation der Implementierung

```bash
python3 selftest.py                      # 11 Unit-Tests der Kristallmathematik
python3 reference_solver_numpy.py --micro micro_roi.npz --tag roi   # -> verification/
python3 compare_check.py roi             # dolfinx-Ergebnis vs. Pruefrechnung
```

Bereits einmal durchgefuehrt (Fall `roi`, s=1): dolfinx
E_apparent = 182,865200238 GPa gegen numpy-Pruefrechnung 182,865200236 GPa,
relative Abweichung 1,3e-11; Zonenmoduln 187,037 / 175,194 / 185,963 GPa in
beiden identisch. Die Implementierung ist damit bestaetigt; die
Verifikationsdateien wurden anschliessend entfernt.

## Was der Vorfaktor s(x) macht

In der Uebergangszone (zone==1) wird der lokal rotierte, auf den ebenen
Spannungszustand kondensierte Steifigkeitstensor jeder Zelle mit einem
skalaren Faktor multipliziert:

    C_Zelle(x) = s(x) * C_kondensiert(Orientierung, Phase)

Das aendert nur die *Groesse* der Steifigkeit, nicht ihre Anisotropie-Richtung
(alle Eintraege werden gleich skaliert; s=1 ist der reine Mikrostruktur-Fall).
`--sfun` nimmt einen beliebigen numpy-Ausdruck in der Lastachsen-Koordinate
x [µm] entgegen, ausgewertet am Zellmittelpunkt:

| Aufruf | Bedeutung |
|---|---|
| `--sfun "1.0"` | Default: nichts skaliert, nur gemessene Mikrostruktur |
| `--sfun "1.33"` | konstante Ueberhoehung in der ganzen Uebergangszone |
| `--sfun "1 + 0.50*np.exp(-((x-1050.)/350.)**2)"` | **Gauss-Profil**: Ueberhoehung maximal in der Mitte der Uebergangszone (x=1050 µm im ROI), klingt mit sigma=350 µm zu beiden Seiten ab |
| `--sfun "1 + 0.6*(x-636.)/828."` | linearer Anstieg ueber die Zone |

Der Gauss-Fall ist gedacht als physikalisch plausiblere Variante der
konstanten Ueberhoehung: falls die Versteifung aus einem
Diffusions-/Mischungsprofil (Cu-, Ni-, Cr-Gradient, Ausscheidungen) stammt,
ist sie in der Mitte der Mischzone am staerksten und faellt zu beiden Seiten
zu den Monobereichen hin ab - statt an den Zonengrenzen zu springen. Mit
Amplitude 0,50 und sigma=350 µm trifft das Zonenmittel den Messwert
(232,0 vs. 232,4 GPa), erzeugt aber lokal eine Spitze von ~272 GPa. Damit
laesst sich spaeter ein gemessenes Profil (DIC) punktweise nachfahren, statt
nur den Zonenmittelwert zu treffen.

## Zum Vergleich mit dem DIC-Verlauf (Abbildung, Panel d/e)

Der simulierte Ausschnitt ist ~2,8 mm breit, die DIC-Messstrecke umfasst die
ganze Probe (14 Messpunkte). Beide werden daher **getrennt** dargestellt:
Panel (d) zeigt das Modellprofil nur ueber das simulierte Fenster mit den
experimentellen Zonenmittelwerten als Baender; Panel (e) zeigt den vollen
DIC-Verlauf mit dem simulierten Fenster markiert. Die genaue Zuordnung
EBSD-ROI <-> DIC-Messpunkt-Index ist nicht bekannt (die markierte Lage in
Panel (e) ist eine Annahme: Peak = Grenzflaeche).

## Erste Ergebnisse (dolfinx, Platzhalterkonstanten, 2026-08-20)

ROI/Zonen aus `WAAM_N=1_A12D_Uebergangsbereich_mit_AR_Bereich.bmp`
(schwarzes Rechteck = ROI x∈[30,2811]µm, y∈[1339,2225]µm; grünes Rechteck =
Übergangszone x∈[667,1495]µm). Aufbaurichtung = −x (Pfeil), Zugprobe V ⇒
Lastachse = x. Gitter 263×825 = 216 975 Zellen (1 Zelle = 1 EBSD-Pixel,
3.371 µm); Netzkonvergenz geprüft (Halbierung der Auflösung: ΔE < 0.5 %).

| Zone | E_lokal Experiment (DIC) [GPa] | Modell s=1 | Modell s=1.33 | Modell s(x) Gauß |
|---|---|---|---|---|
| 17-4PH | 201.7 ± 2 | 187.0 | 187.5 | 187.3 |
| Übergang/Grenzfläche | **232.4 ± 13** | **175.2** | **232.4** | **232.0** |
| 316L | 162.7 ± 2 | 186.0 | 186.0 | 186.0 |
| global (ROI) | 212.4 ± 2 (ganze Probe) | 182.9 | 198.1 | 198.0 |

(Werte aus `E_<tag>.json`; der Fall s=1 ist gegen die numpy-Pruefrechnung
verifiziert, s. oben. Die beiden s-Varianten bitte mit `bash run_fem.sh` neu
rechnen, falls die Dateien fehlen.)

Befund: Mit der gemessenen Kornstruktur/Orientierung allein (s=1) ist der
Übergangsbereich im Modell die *weichste* Zone — die experimentelle
Steifigkeitsüberhöhung wird **nicht** durch die Mikrostruktur des Schliffs
erklärt. Ein konstanter Vorfaktor s≈1.33 auf den C-Tensor der Übergangszone
reproduziert den Messwert exakt (Stellschraube `--sfun`). Diskussionspunkte:
Platzhalter-Einkristallkonstanten, ebener Spannungszustand (keine
3D-Constraints), nur ein Schliff, DIC-Fenster ≠ EBSD-Zonen, mögliche
Zusatzphasen/Ausscheidungen im Übergang.
