# Restart nach SLURM-Timeout (075 2D und 076 3D)

Stand: 19.10.2026. Gilt für `075_klaim_rev2_2D` **und** `076_klaim_rev2_3D` (gleiche Mechanik, gleiche Dateinamen,
`klaim_restart.py` und `11_resubmit_timeouts.sh` sind in beiden Ordnern identisch). Vorbild war der Restart der
Fließflächenstudien `Meshing/pygalmesh/data/scripts/014-Yield-Surface-From-leS` / `017-…` (`yield_restart.py`),
insbesondere die partitionsunabhängige Zuordnung (die PT-SCOTCH-Nummerierung ist nicht reproduzierbar).

## Kurzfassung

* `run_simulation.py` schreibt alle `CHECKPOINT_MINUTES` Minuten Wandzeit (Default 30) und nach dem letzten Schritt
  einen Checkpoint `run_simulation_checkpoint.h5` (+ lesbare Kopie `run_simulation_checkpoint.json`).
* Das Jobskript ruft `run_simulation.py --restart auto` auf. Liegt ein gültiger Checkpoint im Fallordner, wird dort
  fortgesetzt (Zeit, dt, K-Feld-Position, alle kumulierten Größen); sonst startet die Rechnung frisch **genau wie bisher**.
  Bei vorhandenen Restart-Daten werden Netz und Homogenisierung im Jobskript übersprungen.
* `./11_resubmit_timeouts.sh` findet abgelaufene Fälle und reicht sie erneut ein.
* Die Prozesszahl darf sich beim Restart ändern (getestet: 4 → 3 Prozesse, siehe unten).
* Physik, Solver-Parameter und `alex/solution.py` sind unverändert.

## Bedienung auf dem Cluster

```bash
# 1. Code holen (Mac: commit + push, siehe Git-Befehle im CLAUDE.md-Eintrag vom 19.10.2026)
cd $HOME/dolfinx_alex/shared/scripts && git pull
chmod +x 075_klaim_rev2_2D/*.sh 076_klaim_rev2_3D/*.sh

# 2. Was würde neu eingereicht?  (laufende/wartende Jobs werden nie angefasst)
cd $HOME/dolfinx_alex/shared/scripts/075_klaim_rev2_2D
./11_resubmit_timeouts.sh --dry-run                 # alle 2D-Fälle
./11_resubmit_timeouts.sh --dry-run --match S1_vM   # nur eine Auswahl

# 3. Einreichen
./11_resubmit_timeouts.sh [--match TEXT] [--max N]
cd ../076_klaim_rev2_3D && ./11_resubmit_timeouts.sh --dry-run && ./11_resubmit_timeouts.sh

# 4. Kontrolle (im .out des neuen Jobs)
grep -h "RESTART\|CHECKPOINT" $HPC_SCRATCH/075_klaim_rev2_2D/<fall>/*.out.* | tail
./07_status.sh --match <fall>
```

Auswahlregel von `11_resubmit_timeouts.sh` je Fallordner: Jobname (`#SBATCH -J` aus `job_script.sh`) nicht in `squeue`,
neueste `*.out.*` ohne `finished with exit code 0`, und entweder letzter `sacct`-Status `TIMEOUT`, oder
`DUE TO TIME LIMIT` in der neuesten `*.err.*`, oder ein nicht als fertig markierter Checkpoint (z. B. nach NODE_FAIL).
Ordner ohne `*.out.*` (nie gelaufen) bleiben `03_submit_all.sh` überlassen; abgestürzte Läufe (Exit-Code ≠ 0, ohne
Checkpoint) werden nur gelistet — erst die Ursache prüfen. Eingereicht wird mit `sbatch -A $ACCOUNT --exclude=$EXCLUDE
job_script.sh` (Default `p0023647` und die Ausschlussliste aus `02_create_jobs.sh`, kein `-C`;
`OPENBLAS_CORETYPE=SkylakeX` setzt `job_script.sh`). Anderes Konto: `ACCOUNT=special00008 ./11_resubmit_timeouts.sh`.
Bei `AssocMaxSubmitJobLimit` bricht das Skript mit Exit-Code 3 ab; später einfach erneut aufrufen.

**Alte Fallordner** (vor dem 19.10.2026 angelegt, also alle derzeit laufenden Jobs) werden vor dem Einreichen
aufgerüstet: `run_simulation.py` und `klaim_restart.py` kommen aus `000_template` (das alte Skript bleibt als
`run_simulation.py.pre_restart`), `job_script.sh` wird mit `02_create_jobs.sh --exact <ordner>` neu geschrieben (mit den
*aktuellen* globalen Einstellungen von `02_create_jobs.sh`, u. a. `TIME`). Da diese Läufe noch keinen Checkpoint haben,
setzen sie an der **letzten Feldausgabe** (`run_simulation.xdmf/.h5`, alle `POSTPROCESSING_INTERVAL` = 400 (2D) / 200 (3D)
Schritte) fort — siehe „Legacy-Restart" unten.

Weitere Schalter (Umgebung beim `sbatch`, werden ins Jobskript durchgereicht):

| Variable | Wirkung |
|---|---|
| `CHECKPOINT_MINUTES=60` | anderes Checkpoint-Intervall (0 = nur nach dem letzten Schritt) |
| `RESTART_MODE=no` | erzwingt einen frischen Start (alte Tabellen und Felder werden überschrieben!) |

Ein frischer Neustart eines Falls geht also mit `RESTART_MODE=no sbatch job_script.sh` oder durch Löschen von
`run_simulation_checkpoint.*` **und** der Feldausgabe `run_simulation*.xdmf/.h5`. Achtung: Das alte Rezept „`*.out.*`
löschen und `03_submit_all.sh`" setzt jetzt fort statt neu zu starten, sobald Restart-Daten im Ordner liegen.

## Was im Checkpoint steht

`run_simulation_checkpoint.h5` (eine Datei, von Rang 0 geschrieben: erst `….h5.tmp`, dann `os.replace` → ein Abbruch
während des Schreibens lässt den vorigen Checkpoint intakt; die Metadaten stehen als Attribut in derselben Datei, es
kann also keine Feld-/Metadaten-Mischung verschiedener Zeitpunkte geben):

| Gruppe | Inhalt |
|---|---|
| `p1/w/sub0`, `p1/w/sub1`, `p1/wm1/sub*` | Verschiebung u und Phasenfeld s (aktueller und vorheriger Zustand) je eigenem Freiheitsgrad, mit `original_index` (Knotennummer in der Netzdatei) und Knotenkoordinaten `x` |
| `quad/alpha_n`, `quad/alpha_tmp`, `quad/e_p_*` | interne Variablen am (einen) Quadraturpunkt je Zelle: 2D `e_p_11/22/12/33_n` und `_n_tmp`; 3D `e_p_11/22/33/12/13/23_n` und `_prev` |
| `cells/original_index`, `cells/midpoint` | Zellnummer in der Netzdatei und Zellmittelpunkt (= Quadraturpunkt bei `deg_quad = 1`) |
| Attribut `meta` (JSON) | `t` und `dt` des nächsten Schritts, `trestart`, `step`, `Work`, `D_visc`, `D_pl_deg`, `D_pl_undeg`, `success_timestep_counter`, das `state`-Dict (`x_ct_prev`, `t_prev`, `Jx_prev`, `n_event_snapshots`, `last_event_step`, …), `event_snapshots`, Anzahl Tabellenzeilen und Dateigrößen der Tabellen, Restart-Generation, Fall-Fingerabdruck (Material, σ_y, H, β, η, J_c, K_bc, Netzdatei, …), `finished` |

Der Checkpoint entsteht im `before_each_timestep`-Hook, d. h. nach einem akzeptierten Schritt, wenn `t` und `dt` des
nächsten Versuchs schon feststehen — genau der Zustand, mit dem `alex/solution.py` weiterrechnen würde. Die K-Feld-Position
folgt aus `t` (`x_bc = x_0 + v t`). Gehört ein Checkpoint zu anderen Fallparametern (Fingerabdruck), bricht der Lauf ab,
statt falsch fortzusetzen. Ist er als `finished` markiert, beendet sich `run_simulation.py` sofort mit Exit-Code 0.

**Partitionsunabhängige Zuordnung:** Schlüssel sind die Knoten- und Zellnummern der *Netzdatei*
(`geometry.input_global_indices`, `topology.original_cell_index`), die dolfinx bei jeder Prozesszahl gleich liefert. Die
gespeicherten Koordinaten werden zur Kontrolle verglichen (Toleranz 1e-10); passen die Nummern nicht (z. B. Netz neu
erzeugt), wird über die Koordinaten zugeordnet (bitgenau, sonst quantisiert wie in `yield_restart.py`). Fehlt danach
ein Knoten oder eine Zelle (anderes Netz), bricht der Lauf kollektiv mit `RestartError` ab.

## Was beim Restart mit den Ausgaben passiert

* `run_simulation_graphs.txt`, `run_simulation_log.txt` (Newton-Log) und `run_simulation_events.txt` werden
  **fortgeschrieben**: zuerst auf die Größe zum Checkpoint-Zeitpunkt abgeschnitten (Zeilen, die der abgebrochene Lauf
  danach noch geschrieben hat, fallen weg), dann geht es dort weiter. Die Zeilenzahl wird gegen den Checkpoint geprüft.
  Im Newton-Log und in der Event-Datei markiert eine `#`-Zeile den Restart. Die Tabelle selbst bleibt kommentarfrei.
* Feldausgabe in eine neue Datei: `run_simulation_r1.xdmf` (dann `_r2`, …), Event-Snapshots in
  `run_simulation_events_r1.xdmf`; die Snapshot-Indizes laufen weiter (die Kommentarzeile in
  `run_simulation_events.txt` sagt, ab welchem Index die neue Datei gilt).
* `parameters.txt` wird beim Restart nicht erneut ergänzt (gleiche Werte wie beim ersten Lauf).
* Die Kopie in `simulation_<timestamp>/` am Ende jedes Jobs bleibt wie bisher.

## Legacy-Restart (Läufe ohne Checkpoint)

Für Ordner ohne Checkpoint, aber mit Feldausgabe, liest `--restart auto` den letzten vollständigen Zeitschritt aus
`run_simulation.xdmf/.h5` (u, s, alpha, eps_p; DG0 ist bei `deg_quad = 1` exakt, 2D: `e_p_33 = -(e_p_11 + e_p_22)`) und
ordnet Knoten über die Koordinaten und Zellen über die sortierten Knoten-Tupel zu (wie `yield_restart.py`). `t`, `dt`
und die Iterationszahl kommen aus dem Newton-Log (volle Genauigkeit), das nächste `dt` wird wie in `alex/solution.py`
bestimmt. Die kumulierten Größen `W`, `D_visc`, `D_pl_deg`, `D_pl_undeg` und `x_ct`, `J_x` für die Sprungerkennung kommen
aus der Tabellenzeile zu diesem Zeitpunkt — die Tabelle hat nur **4 Nachkommastellen** (`%.4e`), die fortgesetzten
kumulierten Größen tragen also einen konstanten Versatz von bis zu 5e-5 relativ. Verloren sind höchstens die
`POSTPROCESSING_INTERVAL` Schritte seit der letzten Feldausgabe. Der Legacy-Weg wird nur benutzt, wenn kein Checkpoint
existiert; jeder fortgesetzte Lauf schreibt danach eigene Checkpoints.

## Grenzen

* Gleiches Netz nötig (wird geprüft). Gleiche Physik nötig (Fingerabdruck wird geprüft).
* Ein Restart reproduziert den Lauf nicht bitgenau: andere Prozesszahl → andere Rundung in MUMPS. Die adaptive
  Schrittweitensteuerung kann dadurch an Grenzfällen (3 vs. 4 Newton-Iterationen) anders entscheiden; die Ergebnisse
  stimmen dann im Rahmen der Lösertoleranz überein, die Zeitpunkte der Tabellenzeilen aber nicht mehr eins zu eins.
* Verloren geht bei einem Timeout die Rechenzeit seit dem letzten Checkpoint (≤ `CHECKPOINT_MINUTES`). SIGTERM wird
  nicht abgefangen (ein kollektiver Schreibvorgang im Signal-Handler ist unter MPI riskant).
* Kosten: Checkpoint 2D ≈ 0,1 s (150k Freiheitsgrade, 4 Prozesse). Dateigröße 2D ≈ 20 MB (150k Freiheitsgrade, 96k Zellen), 3D entsprechend größer.

## Test (lokal, Docker `dolfinx_alex-alex-dolfinx`, 19.10.2026)

Testordner (nicht in git): `075_klaim_rev2_2D/results_local/restart_test/` (`run.sh`, `compare2.py`),
`076_klaim_rev2_3D/results_local/t3d/`.

**2D, w_s = 1.0, vM, K_bc_rule equal_gc, STOP_FRACTION = 0.25** (151 386 Freiheitsgrade):
(a) Referenz ohne Unterbrechung, 4 Prozesse (1678 Schritte, bis Separation x_ct = 4,0 bei t = 5,7553);
(b) gleicher Lauf mit `CHECKPOINT_MINUTES=2`, 4 Prozesse, nach dem Checkpoint bei Schritt 352 (t = 1,2221, x_ct = 2,32)
mit `kill -9` bei Zeile 372 abgebrochen, dann `--restart auto` mit **3** Prozessen bis zum Ende (1283 Zeilen).
Die 20 Zeilen nach dem Checkpoint wurden korrekt abgeschnitten, Feldausgabe ging nach `run_simulation_r1.xdmf`.

| Größe | Referenz (Ende) | Restart (Ende) | rel. Abw. Ende | max. Abw. stabile Phase nach Restart (über t) | max. Abw. Sprungphase (über x_ct) |
|---|---|---|---|---|---|
| t | 5,7553 | 5,7555 | 3,5e-5 | | |
| J_x | 1,2246 | 1,2244 | 1,3e-4 | 1,6e-3 | 2,6e-3 |
| x_ct | 4 | 4 | 0 | 3,9e-3 (knotenweise, Δh = 0,02) | |
| W | 14,73 | 14,73 | < 7e-5 | 6,7e-5 | 6,7e-5 |
| E_el | 6,3417 | 6,3407 | 1,2e-4 | 1,5e-3 | 2,4e-3 |
| E_pl | 0,93179 | 0,93175 | 4,3e-5 | 4,5e-3 | 1,0e-2 |
| D_visc | 148,54 | 148,54 | < 7e-5 | 0 | 6,7e-5 |
| D_pl_deg | 2,1240 | 2,1244 | 1,9e-4 | 4,2e-3 | 7,6e-3 |
| A | 3,6265 | 3,6272 | 2,2e-5 | 2,3e-4 | 2,3e-4 |

(relativ zum Maximalbetrag der Größe in der Referenz; Tabellenauflösung `%.4e` ≈ 5e-5.)
Wichtig zur Einordnung: Zwei **ununterbrochene** Läufe mit derselben Prozesszahl sind hier schon nicht identisch — (a)
und der erste Abschnitt von (b) (beide 4 Prozesse) haben ab Schritt 26 verschiedene Schrittfolgen (PT-SCOTCH-Partition
und MUMPS sind nicht deterministisch, die adaptive dt-Steuerung verstärkt Rundungsunterschiede). Im Abschnitt vor dem
Restart waren die Abweichungen (a)–(b) auf dem t-Raster sogar größer (J_x 1,8e-2, E_pl 5e-2 relativ, im steilen
Anfangstransienten). Die Abweichungen nach dem Restart liegen also im Rahmen dieser Lauf-zu-Lauf-Streuung; die
Endwerte stimmen auf ≤ 2e-4 überein. Die Sprungphase (x_ct 2,5 → 4 innerhalb Δt ≈ 2e-4) ist über x_ct verglichen,
weil eine Verschiebung des Sprungzeitpunkts um 2e-4 auf dem t-Raster scheinbar große Differenzen erzeugt.

**Zuordnung bitgenau:** Checkpoint von 4 Prozessen mit 2 bzw. 4 Prozessen geladen und sofort neu geschrieben → alle
Felder (w, wm1, alpha, e_p, …) über die Original-Indizes verglichen **bitidentisch**.

**Legacy-Restart (2D):** Lauf mit `POSTPROCESSING_INTERVAL=50`, 2 Prozesse, ohne Checkpoint, bei Zeile 218 abgebrochen,
mit 3 Prozessen aus der Feldausgabe bei Schritt 200 fortgesetzt bis zum Ende (1809 Zeilen). Endwerte gegen
die Referenz: J_x 1,2246 / 1,2246, W 14,73 / 14,73, E_el 1e-5, E_pl 1,8e-4, D_pl_deg 6e-4, A 3e-5 relativ, t_Ende 5,7559 statt
5,7553; in der Sprungphase (über x_ct) J_x 1e-3, E_pl 3e-3 relativ. Dieser Lauf hatte vor dem Abbruch eine andere
Prozesszahl (2) als die Referenz, daher schon vor dem Restart andere Schrittfolgen.
Ein Restart mit geänderter Physik (`--beta 0.21`) wird mit `RestartError` abgelehnt, die Tabellen bleiben unberührt.

**3D, w_s = 1.0, vM** (137 708 Freiheitsgrade, `NL = 5`): 4 Prozesse mit `CHECKPOINT_MINUTES=1`, nach dem Checkpoint bei
Schritt 7 abgebrochen (8 Zeilen), Restart mit 3 Prozessen: Tabelle bei Zeile 7 fortgesetzt, die neu gerechnete Zeile 8
ist in allen Spalten (`%.4e`) identisch mit der des abgebrochenen Laufs; weitere Schritte und Checkpoints laufen.
Zweiter Test mit `STOP_FRACTION=0.001`: Lauf endet nach dem ersten Schritt, schreibt den finalen Checkpoint
(`finished: true`), ein erneuter Aufruf mit `--restart auto` beendet sich sofort mit Exit-Code 0.

**`11_resubmit_timeouts.sh`** mit gefälschten `squeue`/`sacct`/`sbatch` (Linux-Container) getestet: TIMEOUT-Ordner
alter Bauart → aufgerüstet und eingereicht; NODE_FAIL mit Checkpoint → eingereicht; fertig / in squeue / abgestürzt /
nie gelaufen → übersprungen.
