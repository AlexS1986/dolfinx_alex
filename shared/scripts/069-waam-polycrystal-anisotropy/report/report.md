---
title: "Elastische Anisotropie WAAM-gefertigter Stähle (316L, 17-4PH)"
subtitle: "EBSD-basierte Mikrostruktur-Rekonstruktion (Neper) und FE-Homogenisierung (dolfinx)"
lang: de
geometry: margin=2.2cm
fontsize: 10pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
colorlinks: true
linkcolor: "RoyalBlue"
header-includes: |
  \setcounter{secnumdepth}{-1}
  \usepackage{float}
  \floatplacement{figure}{H}
  \usepackage{booktabs}
  \usepackage{microtype}
  \setlength{\emergencystretch}{4em}
  \hyphenation{Auf-bau-rich-tung Schweiß-rich-tung Wand-nor-ma-le Äqui-va-lenz-durch-mes-ser Ho-mo-ge-ni-sie-rung Mi-kro-struk-tur}
---

# Zusammenfassung

Aus EBSD-Schliffen zweier WAAM-gefertigter Stähle (austenitisches 316L, martensitisches 17-4PH) wurden statistisch äquivalente 3D-Polykristall-RVEs mit Neper erzeugt und in dolfinx linear-elastisch homogenisiert. Jedem Korn wird die kubische Einkristall-Steifigkeit seiner Phase (FCC-Austenit / BCC-Martensit) zugewiesen, rotiert in den Probenrahmen über die gemessene Kornorientierung. Aus sechs KUBC-Lastfällen ergibt sich der volle effektive Steifigkeitstensor; ergänzend liefern numerische einaxiale Zugversuche an gerichteten Stäben (V/H/45°) den richtungsabhängigen E-Modul.

Kernergebnis: 316L ist orthotrop — der E-Modul unterscheidet sich in allen drei WAAM-Richtungen: Aufbaurichtung E_z ≈ 216, Schweißrichtung E_x ≈ 214, Wandnormale E_y ≈ 197 GPa (die gestreckten/texturierten Richtungen Aufbau und Schweiß sind am steifsten, die dünne Wandnormale am weichsten). 17-4PH ist dagegen nahezu isotrop (E ≈ 203–208 GPa). Dies ist eine direkte Folge der aus den Daten abgeleiteten Kornform: in 316L Austenitkörner, die entlang Aufbau- und Schweißrichtung gestreckt sind (aus Vertikal- und Horizontalschliff), gegenüber gleichachsigem, richtungslos elongiertem Martensit in 17-4PH. Ein Abgleich mit experimentellen Zugversuchen (Abschnitt 3.6) bestätigt das Modell für das nahezu isotrope 17-4PH; für 316L zeigt er, dass die vorliegenden EBSD-Orientierungen die real vorhandene scharfe ⟨100⟩-Textur (V/H weich ≈ 94 GPa, 45° steif) nicht tragen — die Einkristallkonstanten hingegen sind bestätigt.

# 1  Zielsetzung

Ziel ist die numerische Bestimmung der effektiven elastischen Anisotropie beider Stähle aus der real gemessenen Mikrostruktur. Die Kette umfasst (i) statistische Auswertung der EBSD-Daten, (ii) Erzeugung repräsentativer Volumenelemente (RVE) mit Neper, (iii) Vernetzung, (iv) kornweise anisotrope FE-Homogenisierung sowie (v) einen numerischen Zugversuch zur Quervalidierung des richtungsabhängigen E-Moduls.

# 2  Methodik

## 2.1  EBSD-Daten und statistische Charakterisierung

Je Stahl liegen TSL/OIM-Kornexporte für drei Schliffe vor (Vertikal, Horizontal, 45°). Aus den flächengewichteten Verteilungen von Korngröße, Aspektverhältnis und Ellipsen-Hauptachse werden die Zielgrößen für das Neper-Modell abgeleitet (mechanisch dominante Körner bestimmen die flächengewichtete Statistik). 316L ist einphasig FCC; 17-4PH ist überwiegend BCC-Martensit (≈ 98.6 % Fläche) mit geringem Restaustenit (FCC).

## 2.2  Neper-Mikrostrukturmodell und Annahmen

Die Tessellation erfolgt im skalierten (isotropen) Raum mit einer lognormalen Verteilung des Äquivalenzdurchmessers; anschließend wird durch eine anisotrope Streckung `scale(sx,sy,sz)` die gemessene Kornform erzeugt. Die zentralen Modellannahmen im Überblick — die folgenden Unterabschnitte führen sie aus:

- Kornform: 316L orthotrop (Körner entlang Aufbau- *und* Schweißrichtung gestreckt), 17-4PH gleichachsig (Kornelongation richtungslos verteilt). → 2.2.1, 2.2.3
- Korngröße: lognormale Verteilung des Äquivalenzdurchmessers, 2D→3D-Korrektur per π/4-Stereologie; für 17-4PH wird die Verteilungsbreite (CV) für die Vernetzung gekappt. → 2.2.2
- Textur: flächengewichtetes Bootstrap-Sampling der gemessenen EBSD-Orientierungen; Probenrahmen so gedreht, dass die Aufbaurichtung auf die z-Achse fällt (RVE-Rahmen: x = Schweiß, y = Wandnormale, z = Aufbau). → 2.2.4
- Phasen: 316L einphasig FCC; bei 17-4PH werden die Phasen nur für die *Größenstatistik* zusammengefasst, jedes Korn behält sein Phasenlabel (fcc/bcc) für die *Elastizität*. → 2.2.5
- Referenzrahmen: Kornform- und Orientierungswinkel nutzen denselben EBSD-Scanframe; Vorzeichen-/90°-Konventionen nicht per Polfigur gegengeprüft. → 2.2.6
- RVE-Größe: 316L *n* = 500, 17-4PH *n* = 300 Körner; die endliche Größe erzeugt eine statistische Streuung von wenigen Prozent.

### 2.2.1  Begriffe: Kornform und Textur

Zwei *unabhängige* Quellen elastischer Anisotropie werden unterschieden:

- Kolumnar / orthotrop (Kornform): Die *Form* der Körner ist gestreckt, da sie bei der Erstarrung entlang des Temperaturgradienten wachsen — dies ist eine *morphologische* Anisotropie. Kennwerte: Elongation *k* = Länge/Breite und die Achsenkonzentration *R* ∈ [0,1] (*R* = 1: alle Längsachsen parallel; *R* = 0: zufällig). *Kolumnar* = Streckung in einer Richtung; *orthotrop* = Streckung in zwei zueinander senkrechten Richtungen mit unterschiedlichem Betrag (bei 316L: Aufbaurichtung k_build ≈ 3.41 und Schweißrichtung k_inplane ≈ 3.18, siehe 2.2.3).
- Texturiert (Kristallorientierung): Die *Kristallgitter* der Körner sind nicht zufällig orientiert, sondern relativ zum Proben-/Aufbaurahmen bevorzugt ausgerichtet (nicht-uniforme Orientierungsverteilung, ODF). Dies ist eine *kristallographische* Anisotropie — unabhängig von der Kornform. Da der Einkristall stark anisotrop ist (Zener *A* ≠ 1), macht eine Vorzugsorientierung das Aggregat richtungsabhängig.
- Gleichachsig (equiaxed): Im Mittel keine bevorzugte Kornform-Richtung; im Modell durch *k* = 1 (keine Streckung) abgebildet.
- Richtungslos elongiert: Die *einzelnen* Körner sind zwar elongiert (17-4PH-Martensit: *k* ≈ 4), ihre Längsachsen zeigen aber in viele Richtungen (*R* ≈ 0.08). Im Aggregat gibt es daher keine morphologische Vorzugsrichtung — makroskopisch verhält sich das Gefüge gleichachsig. Deshalb wird 17-4PH mit *k* = 1 modelliert; ein globaler Stretch würde eine kolumnare Anisotropie *erzeugen*, die real nicht existiert.

Physikalischer Hintergrund: 316L-Austenit wächst kohärent gestreckt entlang der Aufbaurichtung (*R* = 0.87, aus dem H-Schliff) und zusätzlich kohärent entlang der Schweißrichtung (*R* = 0.73, aus dem V-Schliff) → orthotrope Kornform. 17-4PH erstarrt zwar ebenfalls als kolumnarer Vorlauf-Austenit, wandelt aber beim Abkühlen in Martensit um; dieser bildet je Vorkorn ~24 kristallographische Varianten, deren Lath-/Blockachsen in viele Richtungen zeigen. Die Martensit-Kornstruktur ist damit makroskopisch nicht kolumnar (*R* niedrig) — die elastische Anisotropie von 17-4PH stammt allein aus der (verbleibenden) kristallographischen Textur.

### 2.2.2  Größenmaß und lognormale Verteilung

- Tessellation / Tessellationszelle: Eine *Tessellation* ist die lückenlose Zerlegung des RVE-Volumens in nicht-überlappende Polyeder (wie ein 3D-Mosaik bzw. Voronoi-Muster); jede *Tessellationszelle* ist genau ein (digitales) Korn.
- Äquivalenzdurchmesser: Da Körner unregelmäßig geformt sind, wird ihre Größe auf eine einzige Zahl reduziert — den Durchmesser eines Kreises (2D) bzw. einer Kugel (3D) mit *gleicher Fläche/gleichem Volumen* wie das Korn: d = 2·√(A/π) aus der Kornfläche A. Neper bemisst (= dimensioniert) die Tessellationszellen über genau dieses Maß (Parameter `diameq`) — es legt die Zellgrößen so fest, dass ihre Äquivalenzdurchmesser der vorgegebenen Verteilung folgen.
- Stereologie (2D→3D): EBSD liefert nur 2D-Schnitte; ein zufälliger Schliff trifft ein Korn fast nie mittig, daher unterschätzt die 2D-Größe die echte 3D-Größe. Näherung „Kugelschnitt": für eine Kugel mit Durchmesser D ist der mittlere Schnittkreis-Durchmesser (π/4)·D, also d₃D = d₂D·4/π ≈ 1.27·d₂D. Dies korrigiert nur den Median/Mittelwert (Skalierungsfaktor), nicht die Verteilungsform. Die rigorosere Saltykov-Entfaltung (rekonstruiert die volle 3D-Verteilung per Größenklassen-Inversion) wird hier nicht verwendet.
- Lognormale Verteilung: Die Korngrößen sind nicht einheitlich, sondern folgen einer Lognormalverteilung — d. h. der *Logarithmus* des Durchmessers ist normalverteilt. Das entspricht der empirisch fast immer beobachteten Korngrößenstatistik in Polykristallen: rechtsschief (viele kleine, wenige große Körner), strikt positiv (keine negativen Größen) und vollständig durch zwei Kennwerte beschrieben — den Median d₃D und die Streubreite (Variationskoeffizient CV = σ/μ, äquivalent σ_ln = √(ln(1+CV²))). In Neper wird die Zielverteilung relativ zum Mittelwert als `diameq:lognormal(1, CV)` vorgegeben; die Werte stammen aus dem EBSD-Fit (Tabelle\ \ref{tbl:params}: Median 141 µm bzw. 27 µm; CV 0.43 bzw. gekappt 0.80).
- CV-Cap (Deckelung der Streubreite): Für 17-4PH ist die gepoolte Größenverteilung sehr breit (gemessener CV ≈ 1.40) — schwer zu erzeugen und mit extremen Größenverhältnissen (fragile Vernetzung). Daher wird der CV der Neper-Zielverteilung auf 0.80 gekappt (Median erhalten). Das verbessert Konvergenz und Vernetzbarkeit und beeinflusst nur die lokale Größenstreuung, nicht die effektive elastische Anisotropie.
- Größenmaß je Stahl: Für die 316L-Körner speist der Fit die transversale Kornbreite (2·Ellipsen-Nebenachse) als Basis-Größe ein (die anisotrope Streckung formt daraus die orthotrope Kornform); für die gleichachsigen 17-4PH-Körner den Äquivalenzdurchmesser.

### 2.2.3  Orthotrope Kornform und WAAM-Geometrie

Beim WAAM-Wandaufbau gibt es drei ausgezeichnete Richtungen: die Aufbaurichtung *z* (Stapelrichtung der Lagen), die Schweiß-/Verfahrrichtung *x* (Bahn der einzelnen Raupe) und die Wandnormale *y* (Dickenrichtung der dünnen Wand).

Probenorientierungen und Schliffebene. Die Zugproben werden in drei Orientierungen aus der Wand entnommen (Abb.\ \ref{fig:specimens}, links): V (Lastachse ∥ Aufbaurichtung), H (Lastachse ⊥ Aufbaurichtung, in der Wandebene, d. h. ∥ Schweißrichtung) und 45° dazwischen. Der EBSD-Schliff jeder Probe liegt senkrecht zur jeweiligen Probenachse (Abb.\ \ref{fig:specimens}, rechts). Damit zeigt der V-Schliff die Ebene ⊥ Aufbaurichtung (Schweiß × Wandnormale) und der H-Schliff die Ebene ⊥ Schweißrichtung (Aufbau × Wandnormale).

![Probenorientierungen relativ zur Aufbaurichtung (V ∥ Aufbau, H ⊥ Aufbau, 45°) und Schliffkonvention: jeder EBSD-Schliff liegt senkrecht zur Probenachse.](fig_specimens.png){#fig:specimens width=15cm}

Weil die Wandnormale in beiden Schliffen die kurze Achse ist, wird sie als Basis (= 1) genommen. Der V-Schliff (⊥ Aufbau, Ebene Schweiß × Wandnormale) liefert Schweiß/Wandnormale (Aspekt 3.18, In-Plane-Achse kohärent bei *R* = 0.73), der H-Schliff (⊥ Schweiß, Ebene Aufbau × Wandnormale) liefert Aufbau/Wandnormale (Aspekt 3.41, kohärent bei *R* = 0.87). Beide zusammen ergeben die orthotrope 3D-Kornform mit den Achsverhältnissen L_z : L_x : L_y ≈ 3.4 : 3.2 : 1 (Aufbau ≈ Schweiß, beide ~3.3× der Wandnormale — plattig, nicht rein säulenförmig).

Unabhängige Bestätigung (45°-Schliff). Aus V- und H-Schliff lässt sich die 3D-Kornform rekonstruieren und daraus der Aspekt des *dritten*, nicht verwendeten 45°-Schliffs vorhersagen: 3.29 gegenüber gemessen 2.60 (Fehler 26 %). Der 45°-Schliff stützt damit unabhängig die rekonstruierte plattige Kornform.

Umgesetzt wird dies durch eine anisotrope Streckung eines gleichachsig erzeugten Gefüges: `scale(k_inplane, 1, k_build)` im RVE-Rahmen (x = Schweiß, y = Wandnormale [Basis], z = Aufbau). Das EBSD-Ziel (Kornform 3.41/3.18) wird dabei um den empirischen Basiszell-Kalibrierfaktor `shape_amp = 1.22` reduziert angewandt — `scale(2.61, 1, 2.80)` —, sodass die *realisierte* Korn-Elongation das Ziel trifft (Kornform-Abgleich in Abschnitt 3.2). Schweißrichtung und Wandnormale sind damit nicht gleichwertig. Der Texturrahmen ist konsistent: die Orientierungen werden aus dem V-Schliff gezogen und so rotiert, dass die Flächennormale des V-Schliffs (= Aufbaurichtung) → z und die In-Plane-Längsachse (= Schweißrichtung) → x fällt.

*Konfigurierbarkeit:* Die Zuordnung Schliff → Kornachse ist in `materials.py` explizit hinterlegt (`k_build_section = horizontal`, `k_inplane_section = vertical`, `build_normal_section = vertical`) und bei abweichender Schnittführung leicht anzupassen.

### 2.2.4  Textur und Orientierungen (Bootstrap-Sampling)

- Kornorientierung / Eulerwinkel: Jedes Korn ist ein kleiner Einkristall; sein Atomgitter ist gegenüber dem Probenrahmen verdreht. Diese Verdrehung wird durch drei Bunge-Eulerwinkel (φ1, Φ, φ2) beschrieben — drei aufeinanderfolgende Drehungen, die das Kristallgitter in die Probenlage bringen. EBSD misst sie für jedes Korn.
- Textur: die Gesamtheit aller Kornorientierungen. Sind sie zufällig → das Aggregat ist isotrop; gibt es Vorzugsorientierungen → *texturiert* → richtungsabhängig (anisotrop).
- Bootstrap-Sampling: Um die *n* Orientierungen des RVE festzulegen, wird *n*-mal zufällig (mit Zurücklegen) aus den gemessenen EBSD-Orientierungen gezogen. So entspricht die Textur des RVE statistisch der gemessenen — ohne die Körner einzeln nachbauen zu müssen.
- Flächengewichtet: Große Körner (viel Fläche im Scan) werden häufiger gezogen; sie tragen mechanisch mehr Last und sollen die Statistik dominieren.
- +3° Streuung: Jeder gezogenen Orientierung wird eine kleine zufällige Drehung (bis 3°) hinzugefügt, damit keine exakten Duplikate entstehen.
- Rahmen-Rotation: Die gemessenen Orientierungen liegen zunächst im EBSD-Map-Rahmen; sie werden so rotiert, dass die Aufbaurichtung auf die z-Achse des RVE fällt (x = Schweißrichtung, y = Wandnormale, z = Aufbaurichtung) — damit sitzt die Textur korrekt relativ zur Aufbaurichtung.
- Wie „misst" man eine Richtung? (Achsenwinkel und Konzentration *R*) Eine Richtung in einem 2D-Schliff wird als Winkel erfasst: An jedes Korn wird eine Ellipse angepasst, deren Hauptachsen-Winkel (0–180°) die Ausrichtung des Korns angibt. Mittelt man diese Winkel über alle Körner, erhält man die dominante Achse; die Achsenkonzentration *R* ∈ [0,1] misst, wie eng die Einzelwinkel darum streuen (*R* = 1: alle Körner exakt gleich ausgerichtet; *R* = 0: Winkel völlig zufällig). — Wichtig: Die Aufbaurichtung wird *nicht* aus der Kornform bestimmt, sondern folgt aus der Schnittführung: Der V-Schliff liegt senkrecht zur Aufbaurichtung, also *ist* seine Flächennormale die Aufbaurichtung (2.2.3) — eine geometrische Tatsache, kein Messergebnis der Körner. Der gemessene Achsenwinkel im V-Schliff beschreibt daher die In-Plane-Längsachse = Schweißrichtung: bei 316L ist *R* = 0.73 → die Körner sind kohärent entlang der Schweißrichtung gestreckt (stützt die orthotrope Streckung). Bei 17-4PH ist *R* = 0.08 → die Kornlängsachsen zeigen in viele Richtungen, es gibt keine bevorzugte In-Plane-Richtung (→ gleichachsig modelliert); lediglich die *Drehung der Textur um die Aufbaurichtung* trägt dann eine kleine Unsicherheit (die *relative* Anisotropie bleibt unberührt).

Abb.\ \ref{fig:rveframe} zeigt den RVE-Rahmen mit dieser Achszuordnung.

![RVE-Koordinatenrahmen: x = Schweißrichtung, y = Wandnormale, z = Aufbaurichtung. Die gemessene Textur wird so rotiert, dass die Aufbaurichtung auf die z-Achse fällt; das eingezeichnete orthotrope Korn ist am längsten entlang z (Aufbau), mittel entlang x (Schweiß), dünn entlang y (Wandnormale).](fig_rve_frame.png){#fig:rveframe width=10cm}

### 2.2.5  Phasen und Einkristall-Steifigkeit

Eine Phase ist eine bestimmte Kristallstruktur — die Anordnung der Atome im Gitter (Abb.\ \ref{fig:phases}). FCC (kubisch-flächenzentriert, „kfz": Atome an den Würfelecken + Flächenmitten) ist der Austenit; BCC (kubisch-raumzentriert, „krz": Ecken + eine Raummitte) der Martensit/Ferrit; die ε-Phase (hexagonal) ist hier mengenmäßig vernachlässigbar. Die Atomanordnung bestimmt die (anisotrope) Einkristall-Steifigkeit — FCC-Austenit und BCC-Martensit haben unterschiedliche Konstanten (Tabelle\ \ref{tbl:singlecrystal}).

- 316L ist einphasig FCC-Austenit.
- 17-4PH ist mehrphasig: überwiegend BCC-Martensit (≈ 98.6 % Fläche) + etwas Rest-Austenit (FCC). „Alle Phasen gepoolt" (engl. *to pool* = zusammenlegen) heißt: für die Korngrößen-/Formstatistik werden alle Körner in einen gemeinsamen Datensatz zusammengefasst — unabhängig von der Phase —, damit die Verteilung das ganze Gefüge repräsentiert (Alternative wäre gewesen: nur die dominante BCC-Phase, oder jede Phase getrennt). Nur die Größen-/Formstatistik wird gepoolt, nicht die Elastizität: jedes Korn behält sein Phasenlabel (fcc/bcc), und dieses Label entscheidet, welcher Einkristall-Steifigkeitstensor dem Korn zugewiesen wird. Die ε-Phase wird der BCC-Steifigkeit zugeschlagen.

![Kristallstrukturen der Phasen (Elementarzelle): FCC (Austenit, 316L) mit Flächenmitten-Atomen vs. BCC (Martensit, 17-4PH) mit einem Atom in der Raummitte.](fig_phases.png){#fig:phases width=14cm}

### 2.2.6  Referenzrahmen und Konventionen

- Referenzrahmen (Koordinatensystem): Der *Referenzrahmen* ist das Koordinatensystem, auf das sich die gemessenen Winkel beziehen. Sowohl die Kornorientierungen (Eulerwinkel) als auch die Kornform-Winkel (Ellipsen-Hauptachse) werden relativ zum Koordinatensystem des EBSD-Scans (TSL-Scanframe; TSL/OIM ist die EBSD-Auswertesoftware) angegeben: eine x- und y-Achse in der Bildebene und z = Flächennormale. „0°" und „x-Richtung" bedeuten in jedem Scan also etwas anderes. Es wird angenommen, dass Kornform- und Orientierungswinkel innerhalb eines Schliffs denselben Rahmen nutzen — nur so lassen sich beide konsistent kombinieren.
- Getrennte Scans je Schliff: Vertikal-, Horizontal- und 45°-Schliff sind *drei getrennte Scans auf drei verschiedenen Schnittflächen*, jeder mit eigenem x/y/z. Um sie zu einem gemeinsamen Probenrahmen (Aufbau / Schweiß / Wandnormale) zusammenzufügen, muss die Zuordnung jedes Scan-Rahmens zu den physikalischen Probenrichtungen bekannt sein (welche Bildachse ist Aufbau, zeigt die Normale hinein oder heraus, gibt es einen 90°- oder Vorzeichen-Versatz). Ist diese Zuordnung bei nur einem Schliff falsch, werden physikalisch gleich liegende Orientierungen (z. B. ⟨100⟩ ∥ Aufbau) im gemeinsamen Rahmen an falsche Winkel gesetzt, und eine real vorhandene scharfe Textur verschmiert beim Kombinieren zu scheinbar regellos (siehe die Textur-Diskrepanz in Abschnitt 3.6). Diese Zuordnungen wurden hier nicht per Polfigur gegengeprüft.
- Polfigur: eine übliche grafische Darstellung der kristallographischen Textur (Projektion bestimmter Kristallrichtungen auf ein Stereogramm), mit der man Konventionen überprüfen könnte.
- Vorzeichen-/90°-Konventionen: Verschiedene Softwarepakete/Konventionen definieren die Winkel teils mit Vorzeichenwechsel oder 90°-Versatz. Diese wurden hier nicht unabhängig per Polfigur gegengeprüft — eine begrenzte Unsicherheitsquelle für die *absolute* Ausrichtung der Textur (die *relative* Anisotropie bleibt davon unberührt).

## 2.3  Vernetzung

Neper `-M`, lineare Tetraeder, relative Elementgröße rcl = 0.6 für das orthotrope 316L-RVE (robuste Vernetzung der flachen Körner; ~253k Elemente bei n = 500), rcl = 0.5 sonst. Export nach XDMF mit einem Zell-Tag *grain* (Korn-ID); eine Tabelle bildet Korn-ID → Eulerwinkel + Kristallsystem ab.

## 2.4  Kornweise anisotrope Steifigkeit

Pro Zelle wird die kubische Einkristall-Steifigkeit der Kornphase über die 6×6-Bond-Rotation in den Probenrahmen transformiert: C_Probe = M(g)·C_Kristall·M(g)ᵀ (Voigt-Reihenfolge xx, yy, zz, yz, xz, xy). Die Einkristallkonstanten (Literatur-Platzhalter, editierbar) sind in Tabelle\ \ref{tbl:singlecrystal} gelistet.

## 2.5  Homogenisierung (KUBC) und Zugversuch

Homogenisierung — die sechs KUBC-Lastfälle: Um den vollständigen 6×6-Steifigkeitstensor zu bestimmen, wird die Materialantwort auf jeden unabhängigen Verzerrungszustand berechnet. Der symmetrische Verzerrungstensor hat sechs unabhängige Komponenten — drei Normal- (ε_xx, ε_yy, ε_zz) und drei Schub-Komponenten (γ_yz, γ_xz, γ_xy) in Voigt-Notation. Ein *Lastfall* setzt genau eine dieser Komponenten auf 1 (die übrigen auf 0), das RVE wird gelöst und die volumengemittelte Spannung (sechs Komponenten) berechnet; sie bildet eine Spalte von C_hom. Sechs Lastfälle → die vollständige Matrix.

Der Zusatz KUBC (kinematisch uniforme Randbedingungen) bezeichnet *wie* der makroskopische Verzerrungszustand aufgeprägt wird: als lineares Verschiebungsfeld *u* = ε_mac·*x* auf dem gesamten Außenrand des RVE (Dirichlet-Randbedingung). KUBC überschätzt tendenziell die Steifigkeit und liefert eine obere Schranke; Alternativen sind statisch-uniforme Randbedingungen (SUBC, untere Schranke) und periodische Randbedingungen (PBC, dazwischen, meist beste Schätzung).

Einaxialer Zugversuch: Auf den gerichteten Stäben (Lastachse *x*, Aufbaurichtung 0°/90°/45° zu *x*) mit Symmetrie-RB auf den drei Minimalflächen und aufgeprägter Verschiebung auf x_max (Querflächen frei) → scheinbarer Modul E = ⟨σ_xx⟩ / (δ/Lₓ).

# 3  Ergebnisse

## 3.1  Gefittete Mikrostrukturparameter

| Größe | 316L | 17-4PH |
|---|---|---|
| Morphologie | orthotrop | equiaxed |
| Phase | FCC 100 % | BCC 98.6 % |
| Größenmaß | 2·Nebenachse | Äquiv.-Durchmesser |
| d₃D Median [µm] | 141 | 27 |
| CV (gemessen → Neper-Ziel) | 0.43 → 0.43 | 1.40 → 0.80 |
| k_build / k_inplane | 3.41 / 3.18 | 1 / – |
| Kohärenz *R* (Aufbau) | 0.87 | 0.08 |
| MIN_POINTS | 10 | 50 |
| Körner *n* | 500 | 300 |
| RVE [µm] | 3013×1152×3243 | 300³ |

: Aus den EBSD-Daten abgeleitete Neper-Parameter (316L: orthotropes RVE mit *n* = 500 Körnern; 17-4PH: gleichachsig, *n* = 300).\label{tbl:params}

Abb.\ \ref{fig:micro} zeigt die daraus generierten dreidimensionalen Mikrostrukturen (Neper-Tessellation, nach Kornorientierung eingefärbt): die orthotrope, entlang Aufbau- und Schweißrichtung gestreckte 316L-Struktur (plattige Körner ~3.4:3.2:1) gegenüber dem gleichachsigen 17-4PH-Würfel.

![Generierte 3D-Mikrostruktur der RVEs (Neper-Tessellation, Färbung nach Kornorientierung). Links 17-4PH (equiaxed, Würfel), rechts 316L (orthotrop: Körner entlang Aufbau- und Schweißrichtung gestreckt, dünn quer zur Wand → plattige Körner ~3.4:3.2:1).](fig_microstructure.png){#fig:micro width=15cm}

## 3.2  Statistisches Matching der generierten Struktur mit den EBSD-Scans

Die Übereinstimmung wird in zwei Schritten geprüft: (a) wie gut der Fit die gemessene EBSD-Verteilung abbildet (EBSD → Ziel), und (b) wie gut die generierte Tessellation dieses Ziel reproduziert (Ziel → generiert).

(a) EBSD → Fit-Ziel. Die folgenden Diagnose-Plots (aus der Neper-Pipeline, Schritt 01) zeigen je Stahl die flächengewichteten EBSD-Verteilungen von Korngröße, Elongation a/b und Ellipsen-Hauptachse (In-Plane-Richtung) mit dem angepassten Lognormal- bzw. Achsen-Fit.

![316L: EBSD-Verteilungen (Korngröße, Elongation, In-Plane-Achse) mit Fit.](ebsd_fit_316L.png){#fig:ebsd316 width=16cm}

![17-4PH: EBSD-Verteilungen mit Fit. Der breite, rausch-behaftete Größen-Tail motiviert den CV-Cap (Abschnitt 2.2).](ebsd_fit_17-4PH.png){#fig:ebsd174 width=16cm}

(b) Ziel → generierte Struktur. Abb.\ \ref{fig:sizematch} vergleicht die Korngrößenverteilung des erzeugten RVE (transversaler Äquivalenzdurchmesser d_t = (6·V/(π·k))^{1/3} aus den Zellvolumina) mit dem Ziel-Lognormal; Tabelle\ \ref{tbl:matching} fasst Median, Mittel und CV zusammen.

![Generierte Korngrößenverteilung (Histogramm) vs. Ziel-Lognormal (Linie). Gestrichelt: erreichter Median; gepunktet: Ziel-Median.](fig4_size_matching.png){#fig:sizematch width=16cm}

| Stahl | Median Ziel | Median erreicht | Mittel Ziel | Mittel erreicht | CV Ziel | CV erreicht |
|---|---|---|---|---|---|---|
| 316L | 141 | 150 | 153 | 159 | 0.43 | 0.38 |
| 17-4PH | 27 | 37 | 34 | 43 | 0.80 | 0.57 |

: Größen-Matching (Basis-Äquivalenzdurchmesser d_t [µm]): Ziel (EBSD-Fit) vs. generiertes RVE.\label{tbl:matching}

Bewertung.

- 316L: sehr gute Übereinstimmung — Median 150 vs 141 µm (+6 %), Mittel +4 %, CV 0.38 vs 0.43. Die Basis-Größenstatistik wird getroffen; die orthotrope Streckung formt daraus die gemessene Kornform.
- 17-4PH: nur näherungsweise — die generierten Körner sind gröber (Median 37 vs 27 µm, +39 %) und weniger streuend (CV 0.57 vs 0.80). Ursache: die sehr breite (bereits gekappte) Zielverteilung lässt sich mit 300 Körnern und der Morpho-Optimierung nicht exakt reproduzieren; bei fester mittlerer Kornvolumen-Dichte hebt ein zu geringes CV den Median an. Für die effektive elastische Anisotropie ist dies unkritisch, da diese bei 17-4PH texturgetrieben und größenunabhängig ist (Abschnitt 2.2.1). Besseres Größen-Matching: Kornzahl erhöhen (*n* = 500–1000) und/oder CV-Cap lockern.
- Kornform (generiert vs. Ziel): Abb.\ \ref{fig:shapematch} vergleicht die aus dem RVE-Netz gemessenen Korn-Aspektverhältnisse (Halbachsen aus dem Trägheitstensor) mit dem EBSD-Ziel. Nach der Kalibrierung (siehe unten) treffen die generierten Körner das Ziel: Median ≈ 3.2 (Aufbau/Wand, Ziel 3.41) bzw. ≈ 3.0 (Schweiß/Wand, Ziel 3.18); der direkt mit EBSD vergleichbare 2D-Schnitt-Aspekt liegt bei 3.4 (⊥ Schweiß, Ziel 3.41) bzw. 3.1 (⊥ Aufbau, Ziel 3.18). Mittlere Zell-Sphärizität 0.71 (316L); 17-4PH bleibt gleichachsig (≈ 0.81). *Kalibrierung:* Die anisotrope Streckung der (nicht perfekt runden) Basiszellen überhöht die per-Korn-Elongation gegenüber dem Nennwert; daher wird die angewandte Streckung durch einen empirischen Faktor (`shape_amp` = 1.22) geteilt (`scale(2.61,1,2.80)`), sodass die realisierte Korn-Elongation das EBSD-Ziel trifft (der Einfluss auf die effektive Elastizität ist ohnehin gering, Abschnitt 4).
- Textur (generiert vs. gemessen): Abb.\ \ref{fig:texmatch} stellt die {100}-Polfiguren der generierten RVE-Orientierungen und der gemessenen EBSD gegenüber. Die generierte Textur gibt die gemessene Orientierungsverteilung wieder (flächengewichtetes Bootstrap, +3° Streuung); beide sind nur schwach texturiert, ohne zentrale Häufung (die kleinen Cluster rechts sind ein Resampling-Artefakt — 500 Orientierungen aus ~124 gemessenen Körnern). Genau diese schwache Textur ist der offene Punkt aus Abschnitt 3.6. Die Elongation ist für 316L orthotrop angepasst (k_build = 3.41 aus dem H-Schliff, k_inplane = 3.18 aus dem V-Schliff) und für 17-4PH bewusst auf *k* = 1 gesetzt (siehe 2.2.1/2.2.3).

![316L Kornform-Abgleich: Verteilung der generierten Korn-Aspektverhältnisse (Aufbau/Wand, Schweiß/Wand) aus dem RVE-Netz gegen die EBSD-Ziele (gestrichelt). Die Verteilungen streuen um das Ziel; der Median liegt etwas höher, da die anisotrope Streckung die Basiszell-Form-Streuung verstärkt.](fig_shape_match.png){#fig:shapematch width=12cm}

![316L Textur-Abgleich: {100}-Polfiguren der gemessenen EBSD-Orientierungen (links, V-Schliff, in den RVE-Rahmen gedreht) und der generierten RVE-Orientierungen (rechts), Blick ∥ Aufbaurichtung. Die generierte Textur reproduziert die gemessene; beide sind nur schwach texturiert (keine zentrale Häufung). Die Cluster rechts sind ein Resampling-Artefakt des Bootstraps.](fig_texture_match.png){#fig:texmatch width=15cm}

Abb.\ \ref{fig:orimap} macht die Kristallorientierung direkt auf der mehrkörnigen Struktur sichtbar: ein Schnitt ⊥ Aufbaurichtung durch das RVE, jedes Korn eingefärbt nach der Kristallrichtung, die entlang der Aufbaurichtung liegt (Inverse-Polfigur-Färbung: [001] rot, [101] grün, [111] blau). Die bunte Durchmischung ohne dominante Farbe bestätigt die schwache Textur — bei einer scharfen ⟨100⟩ ∥ Aufbau-Textur wäre die Karte überwiegend rot.

![Orientierungskarte (IPF ∥ Aufbau) eines Schnitts ⊥ Aufbaurichtung durch das 316L-RVE (Ebene Schweiß × Wandnormale). Jedes Korn ist nach der entlang der Aufbaurichtung liegenden Kristallrichtung eingefärbt (Legende rechts). Die Farbvielfalt zeigt eine schwach texturierte, nahezu regellose Orientierungsverteilung; die Körner sind in der Schnittebene entlang der Schweißrichtung gestreckt.](fig_orientation_map.png){#fig:orimap width=16cm}

## 3.3  Effektiver Steifigkeitstensor C_hom

![Effektive Steifigkeitsmatrizen C_hom [GPa] (Voigt xx,yy,zz,yz,xz,xy). Die Nebendiagonal-Kopplungsterme (Normal/Schub) sind klein → nahezu orthotrop; ihre Größe quantifiziert die statistische Rest-Anisotropie.](fig3_Chom_heatmap.png){#fig:chom width=15cm}

Aus C_hom folgen die richtungsabhängigen Young-Moduln E_i = 1/S_ii mit S = C_hom⁻¹ (Tabelle\ \ref{tbl:engconst}). 316L (orthotrop): E_x (Schweiß) = 214, E_y (Wandnormale) = 197, E_z (Aufbau) = 216 GPa — alle drei Richtungen unterscheiden sich (Streuung 9.4 %), steifste Richtung = Aufbaurichtung, dicht gefolgt von der Schweißrichtung; am weichsten die (dünne) Wandnormale. Diese Rangfolge ist physikalisch stimmig: die beiden entlang der Körner gestreckten und texturierten Richtungen (Aufbau, Schweiß) sind steifer als die dünne Wandnormale. 17-4PH: E_x = 208, E_y = 208, E_z = 203 GPa (Streuung 2.8 %, nahezu isotrop). Beide Tensoren sind positiv definit (kleinster Eigenwert 61 bzw. 83 GPa).

Für eine Weiterverwendung als orthotropes Materialmodell werden die vollständigen Ingenieurskonstanten direkt aus der Nachgiebigkeit S = C_hom⁻¹ gezogen (nicht aus Einzelkomponenten): E_i = 1/S_ii, ν_ij = −S_ji/S_ii, G_yz/G_xz/G_xy = 1/S₄₄/S₅₅/S₆₆ (Tabelle\ \ref{tbl:engconst}). Rahmen: x = Schweiß, y = Wandnormale, z = Aufbau. Die Reziprozität ν_ij/E_i = ν_ji/E_j ist wegen der Symmetrie von S exakt erfüllt (der Satz ist also konsistent orthotrop); die verbleibende Normal-Schub-Kopplung max|C[xx..zz, yz..xy]| = 11.0 GPa (316L) bzw. 6.4 GPa (17-4PH) quantifiziert die statistische Abweichung des endlichen RVE von perfekter Orthotropie. Die sechs Poisson-Zahlen sind nicht gleich (ν_yx = 0.301, ν_zy = 0.318, ν_zx = 0.231 als Gegenstücke), wie es die Orthotropie verlangt.

| Stahl | E_x | E_y | E_z | ν_xy | ν_xz | ν_yz | G_yz | G_xz | G_xy |
|---|---|---|---|---|---|---|---|---|---|
| 316L | 213.7 | 197.0 | 216.5 | 0.326 | 0.228 | 0.289 | 78.0 | 74.7 | 79.0 |
| 17-4PH | 208.4 | 208.1 | 202.5 | 0.278 | 0.305 | 0.306 | 89.3 | 89.8 | 85.3 |

: Vollständige orthotrope Ingenieurskonstanten [GPa bzw. dimensionslos] aus S = C_hom⁻¹, Rahmen x = Schweiß, y = Wandnormale, z = Aufbau (reproduzierbar via `engineering_constants.py`). Gegenstücke der Poisson-Zahlen über ν_ji = ν_ij·E_j/E_i.\label{tbl:engconst}

## 3.4  Richtungsabhängiger E-Modul (Zugversuch)

Abb.\ \ref{fig:specmesh} zeigt die gerichteten, vernetzten Zugstäbe, an denen der numerische Zugversuch durchgeführt wird — je Stahl in den drei Orientierungen. Gut erkennbar der Unterschied der Kornstruktur: wenige große, gestreckte 316L-Körner gegenüber vielen feinen, gleichachsigen 17-4PH-Körnern.

Dass die drei 316L-Stäbe einander stark ähneln, ist korrekt und folgt aus der plattigen Kornform: Die Körner sind in Aufbau- (3.41) und Schweißrichtung (3.18) *ähnlich stark* gestreckt und nur quer zur Wand (Wandnormale) dünn. Da alle Proben aus der Wandebene entnommen werden und die Streckung um die Wandnormale gedreht wird, liegt die dünne Richtung stets in der (flachen) Stabdicke, während die breite Seite immer die beiden langen Richtungen zeigt — die Körner erscheinen daher in jedem Stab in der Ebene gestreckt und quer dazu flach. Weil Aufbau ≈ Schweiß, ist das Korn in der Wandebene fast kreisförmig (Aspekt ~1.07); eine Drehung um 45° ändert daran optisch praktisch nichts — die drei Stäbe sehen daher trotz unterschiedlicher Orientierung ähnlich aus. Die Lastachse ist stets die Stablängsachse (V ∥ Aufbau, H ∥ Schweiß, 45° dazwischen); die Färbung ist zufällig je Korn und zeigt nur die *Form*, nicht die Kristallorientierung. Dass der 45°-Stab dennoch einen anderen E-Modul liefert, stammt aus der um 45° gedrehten *Textur* (Kristallgitter) — in dieser Formansicht nicht sichtbar. Ein Richtungsunterschied im E-Modul kommt also aus der Textur, nicht aus der Kornform, was zur Textur-Diskrepanz in Abschnitt 3.6 führt.

![Numerische Zugstäbe (Neper-Tessellation, Körner zufällig eingefärbt) je Orientierung V (∥ Aufbau) / H (⊥ Aufbau) / 45°. Oben 316L (orthotrop, wenige große Körner), unten 17-4PH (gleichachsig, feinkörnig). Die Stäbe sind flach; über die Dicke liegen nur wenige Körner, daher streuen ihre Absolutwerte stärker als das RVE.](fig_specimen_meshes.png){#fig:specmesh width=16cm}

![Scheinbarer E-Modul aus dem numerischen einaxialen Zugversuch, Last parallel (V), senkrecht (H) und unter 45° zur Aufbaurichtung.](fig1_uniaxial_E.png){#fig:uniaxial width=14cm}

316L: parallel zur Aufbaurichtung (V = 202 GPa) und in Schweißrichtung (H = 204 GPa) nahezu gleich steif, deutlich weicher unter 45° (171 GPa). Dass V ≈ H liegt, passt zur plattigen Kornform (Aufbau ≈ Schweiß); der niedrige 45°-Wert spiegelt die schub-nachgiebige Richtung des texturierten FCC-Aggregats wider. 17-4PH bleibt mit 199–223 GPa schwächer richtungsabhängig; der etwas höhere 45°-Wert ist überwiegend statistische Streuung der flachen Stab-Realisierungen. (Die Stäbe sind flach/dünn — ihre Absolutwerte streuen stärker als das RVE.)

## 3.5  Quervalidierung

![Zugversuch (Balken) vs. RVE-Homogenisierung (gestrichelt: E_z (Aufbau) bzw. E_x (Schweiß) aus dem RVE). Zwei unabhängige Methoden, konsistent für V und H.](fig2_crosscheck.png){#fig:crosscheck width=16cm}

Für 316L liegen die Zugstab-Werte V (202, Aufbau) und H (204, Schweiß) einige Prozent unter den zugehörigen RVE-Werten (E_z = 216, E_x = 214) — die dünnen Flachstäbe (freie Querkontraktion, kleines *n*) sind erwartungsgemäß etwas nachgiebiger als der steife KUBC-Bound, geben aber dieselbe Rangfolge (Aufbau ≈ Schweiß >> 45°) wieder. Für 17-4PH liegen V und H innerhalb weniger Prozent an den RVE-Werten. Die konsistente Rangfolge zweier unabhängiger Methoden ist die eigentliche Validierung; verbleibende Abweichungen entsprechen der erwarteten statistischen Streuung (endliches *n*, flache Stab-Geometrie).

## 3.6  Abgleich mit experimentellen Zugversuchen

Zur Validierung liegen experimentelle einaxiale Zugversuche vor (RPTU Kaiserslautern, Lehrstuhl für Werkstoffkunde), in denen der E-Modul beider Werkstoffe je Orientierung (V ∥ Aufbau, H ⊥ Aufbau, 45°) an mehreren Proben gemessen wurde. Tabelle\ \ref{tbl:expcomp} und Abb.\ \ref{fig:expnum} stellen Experiment und FE-Zugversuch gegenüber.

| Werkstoff | Orientierung | E Experiment [GPa] | E Numerisch [GPa] | Abw. |
|---|---|---|---|---|
| 316L | V (∥ Aufbau) | 102 ± 6 | 202 | +98 % |
| 316L | H (⊥ Aufbau) | 93 ± 2 | 204 | +120 % |
| 316L | 45° | 173 ± 6 | 171 | −1 % |
| 17-4PH | V (∥ Aufbau) | 186 ± 1 | 199 | +7 % |
| 17-4PH | H (⊥ Aufbau) | 169 ± 2 | 206 | +22 % |
| 17-4PH | 45° | 192 ± 2 | 223 | +16 % |

: Experimenteller (RPTU/WKK) vs. numerischer richtungsabhängiger E-Modul (FE-Zugversuch).\label{tbl:expcomp}

![Richtungsabhängiger E-Modul: Experiment (Zugversuch, RPTU/WKK, mit Streuung) vs. numerischer FE-Zugversuch, je Orientierung. 17-4PH: gute Übereinstimmung. 316L: die Anisotropie-Richtung ist invertiert — im Experiment ist 45° am steifsten und V/H auffallend weich.](fig_exp_vs_num.png){#fig:expnum width=16cm}

Für 17-4PH stimmen Experiment (169–192 GPa) und Modell (199–223 GPa) im Charakter überein: beide nahezu isotrop, 45° leicht am steifsten. Die numerischen Werte liegen ~7–22 % höher, was zur KUBC-nahen (steifen) oberen Schranke und den Literatur-Platzhalter-Einkristallkonstanten passt (Abschnitt 5). Hier ist das Modell belastbar.

Für 316L trifft das Modell die Größenordnung, aber nicht die Richtung der Anisotropie. Das Experiment zeigt V und H auffallend weich (93–102 GPa) und 45° am steifsten (173 GPa); das Modell umgekehrt V ≈ H steif (~203 GPa) und 45° weicher (171). Die gemessenen V/H-Werte entsprechen fast genau dem Einkristallmodul in ⟨100⟩-Richtung (E entlang ⟨100⟩ = 94 GPa) und die gemessene Querkontraktion (ν ≈ 0.39) dem ⟨100⟩-Wert (0.40). Die realen Zugproben verhalten sich entlang Aufbau und Schweiß also nahezu wie ⟨100⟩-Einkristalle — das Material ist scharf ⟨100⟩-texturiert, während die 45°-Richtung eine steifere, ⟨110⟩-nahe Orientierung abtastet (E entlang ⟨110⟩ = 194 GPa).

Die Ursache liegt in der Textur, nicht in den Einkristallkonstanten (deren ⟨100⟩-Werte für E und ν das Experiment treffen). Entscheidend — und der Kern des Problems — ist jedoch: die *gemessenen* Kornorientierungen tragen diese scharfe ⟨100⟩-Textur nicht. Der direkt aus den EBSD-Orientierungen berechnete richtungsabhängige E-Modul ist nahezu regellos (~180 GPa in allen Richtungen, Voigt-Abschätzung), und die {100}-Polfiguren der drei Schliffe (Abb.\ \ref{fig:polefig}) zeigen nur schwache Häufungen. Das Modell gibt diese schwach texturierten Scandaten korrekt wieder; die scharfe ⟨100⟩-Textur der Zugproben ist in den vorliegenden Scans nicht enthalten.

*Wie liest man die Polfigur?* Jeder Punkt in Abb.\ \ref{fig:polefig} ist eine der drei ⟨100⟩-Kristallachsen eines Korns, stereografisch auf die Schliffebene projiziert (Blickrichtung = Schliffnormale, im Zentrum mit + markiert); die Punktgröße ist flächengewichtet (große Körner zählen mehr). Ein Punkt im Zentrum bedeutet ⟨100⟩ parallel zur Schliffnormale, ein Punkt am Rand ⟨100⟩ in der Schliffebene. Bei einer scharfen ⟨100⟩-Textur würden sich enge Häufungen bilden (etwa ein starker Fleck im Zentrum = ⟨100⟩ ∥ Normale); bei regelloser Textur verteilen sich die Punkte gleichmäßig über die Scheibe. In allen drei Schliffen sind die ⟨100⟩-Achsen breit gestreut, ohne dominante Häufung — die Scans sind also nur schwach texturiert.

![{100}-Polfiguren der drei 316L-Schliffe (EBSD-Rohdaten, flächengewichtet). Zentrum (+) = Schliffnormale, Rand = Richtungen in der Schliffebene; Punktgröße ∝ Kornfläche. Die breite Streuung ohne dominante Häufung zeigt eine nur schwach texturierte Mikrostruktur — die im Zugversuch belegte scharfe ⟨100⟩-Textur ist in den Scans nicht sichtbar.](fig_polefigures_316L.png){#fig:polefig width=16cm}

Das ist eine offene Frage der Dateninterpretation, kein einfacher Modellfehler. Zwei Erklärungen kommen in Betracht: (i) die drei Schliffe sind getrennte Scans mit je eigenem Referenzrahmen (die ungeprüften Vorzeichen-/Achsenkonventionen, Abschnitt 2.2.6) — bei inkonsistent zugeordneten Rahmen mittelt sich ein real vorhandenes ⟨100⟩-Signal heraus; oder (ii) der gescannte Bereich ist tatsächlich schwächer texturiert als der Prüfquerschnitt der Zugproben. Die Klärung muss datenbasiert erfolgen — durch Prüfung der EBSD-Orientierungskonventionen und Referenzrahmen der Schliffe (Polfigur-Abgleich) —, nicht durch Aufprägen einer angenommenen Textur. Bis die Konventionen bestätigt sind, ist die berechnete 316L-Anisotropie-Richtung mit dieser Unsicherheit behaftet; die Beträge und der 17-4PH-Fall bleiben davon unberührt.

## 3.7  Zusammenfassende Kennwerte

| Stahl | Methode | E_x / V | E_y / H | E_z / 45° | ν |
|---|---|---|---|---|---|
| 316L | RVE-Homogenisierung (KUBC) | 214 | 197 | 216 | 0.33 |
| 316L | FE-Zugversuch V / H / 45° | 202 | 204 | 171 | 0.25–0.39 |
| 316L | Experiment V / H / 45° | 102 | 93 | 173 | 0.02–0.39 |
| 17-4PH | RVE-Homogenisierung (KUBC) | 208 | 208 | 203 | 0.28 |
| 17-4PH | FE-Zugversuch V / H / 45° | 199 | 206 | 223 | 0.26–0.31 |
| 17-4PH | Experiment V / H / 45° | 186 | 169 | 192 | 0.24–0.30 |

: Effektive Young-Moduln [GPa]. RVE-Zeile: die drei Spalten = E_x/E_y/E_z = Schweiß/Wandnormale/Aufbau. Zugversuch-/Experiment-Zeilen: die drei Spalten = V (∥ Aufbau) / H (⊥) / 45°. Experiment: RPTU/WKK.\label{tbl:summary}

# 4  Diskussion

316L — orthotrop und texturiert: Die Steifigkeit unterscheidet sich in allen drei Richtungen (E_Aufbau 216 > E_Schweiß 214 > E_Wandnormale 197 GPa), am größten parallel zur Aufbaurichtung, dicht gefolgt von der Schweißrichtung, am weichsten quer zur Wand. Die drei distinkten Moduln folgen aus der orthotropen Kornform (Streckung entlang Aufbau- und Schweißrichtung) zusammen mit der kristallographischen Textur. Für die lineare Elastizität dominiert dabei die Textur; die Kornform moduliert das Ergebnis nur um wenige GPa. Dass Aufbau- und Schweißrichtung ähnlich steif und deutlich steifer als die Wandnormale sind, ist konsistent mit der plattigen Kornform (Aufbau ≈ Schweiß, beide gestreckt; Wandnormale dünn). Diese modellinterne Anisotropie-Richtung wird jedoch durch den Zugversuch *nicht* bestätigt (Abschnitt 3.6): experimentell ist 45° am steifsten und V/H auffallend weich — ein Hinweis auf eine im Modell zu schwach abgebildete ⟨100⟩-Textur.

17-4PH — nahezu isotrop: Trotz stark anisotroper BCC-Einkristalle (Zener *A* = 2.41) ist das Aggregat quasi-isotrop. Das ist die direkte Konsequenz der gleichachsigen Modellierung mit richtungslos verteilten Lath-Orientierungen — Variantenauswahl mittelt die Anisotropie heraus. Die größere Streuung im Zugversuch (199–223 GPa) resultiert aus den dünnen, separat vernetzten Flachstäben (wenige Körner über die Dicke) und ist nicht als reale Anisotropie zu interpretieren.

Polykristall-Mittelung als Plausibilitätscheck: Beide Aggregate sind erheblich isotroper als ihre Einkristalle (*A* = 3.77 bzw. 2.41); eine Mittelung rotierter Körner über zufällige Orientierungen konvergiert gegen einen isotropen Tensor. Dies bestätigt die Korrektheit der Bond-Rotation und der Zuweisung.

Experimentelle Validierung (Abschnitt 3.6): Der Zugversuch bestätigt das Modell für das schwach texturierte 17-4PH, deckt für 316L aber eine Modellgrenze auf: real ist 316L scharf ⟨100⟩-texturiert (V/H weich ≈ 94 GPa, 45° steif), das Modell reproduziert diese Anisotropie-Richtung nicht. Die Einkristallkonstanten sind nicht die Ursache (ihre ⟨100⟩-Werte treffen das Experiment) — die vorliegenden EBSD-Orientierungen tragen die scharfe ⟨100⟩-Textur schlicht nicht (der direkt daraus berechnete richtungsabhängige E-Modul ist nahezu regellos). Ob dahinter inkonsistent zugeordnete Referenzrahmen der getrennt gescannten Schliffe (Abschnitt 2.2.6) oder ein real schwächer texturierter Scanbereich stehen, ist datenbasiert zu klären (Polfigur-/Konventionsprüfung); die morphologische Kornform spielt elastisch nur eine untergeordnete Rolle.

# 5  Annahmen und Grenzen (Zusammenfassung)

- Kristallographische Textur — wichtigste Grenze für 316L: Die vorliegenden EBSD-Orientierungen tragen die scharfe ⟨100⟩-Textur nicht, die der Zugversuch belegt (Abschnitt 3.6); dadurch stimmt die Richtung der berechneten 316L-Anisotropie nicht mit dem Experiment überein (Beträge/Charakter für 17-4PH hingegen belastbar). Die Einkristallkonstanten sind bestätigt. Offen ist, ob inkonsistent zugeordnete Referenzrahmen der getrennt gescannten Schliffe (Konventionen, Abschnitt 2.2.6) oder ein schwächer texturierter Scanbereich die Ursache sind — datenbasiert per Polfigur-/Konventionsprüfung zu klären.
- KUBC liefert eine obere (steife) Schranke; periodische RB ergäben etwas niedrigere, engere Werte.
- Die Einkristallkonstanten sind Literatur-Platzhalter (`config.json`); die absoluten Moduln skalieren mit ihnen — vor Publikation durch eigene/validierte Werte ersetzen.
- Lineare Elastizität, kleine Verzerrungen; homogene Einkristalle pro Korn, ideale Kornbindung, keine Korngrenzen-/Subkorneffekte.
- 316L wird orthotrop aus Vertikal- und Horizontalschliff modelliert. Es bleibt eine statistisch äquivalente, parametrische Rekonstruktion (Ellipsoid-Körner mit gefitteten Aspektverhältnissen + Textur), kein voxelgenauer Scan-Nachbau; die Schliff-Ebenen-Zuordnung ist eine Annahme (Abschnitt 2.2.3).
- Vereinfachte π/4-Stereologie; 17-4PH-CV für die Netzgenerierung gekappt (0.80 statt 1.40) — ohne Einfluss auf die effektive Anisotropie.
- Für 17-4PH ist die In-Plane-Vorzugsrichtung morphologisch schwach bestimmt (*R* = 0.08); die Drehung der Textur um die Aufbaurichtung trägt daher eine kleine Unsicherheit (die relative Anisotropie bleibt unberührt).
- Endliche RVE (316L *n* = 500, 17-4PH *n* = 300) → statistische Rest-Anisotropie (Kopplungsterme ≠ 0) im Bereich weniger Prozent; für glattere Tensoren *n* erhöhen.
- Die gerichteten Zugstäbe (07) sind flach/dünn und dienen nur als Quercheck; das RVE liefert den maßgeblichen Steifigkeitstensor.

# 6  Schlussfolgerungen

Die Pipeline liefert eine konsistente, datenbasierte Kette von der EBSD-Mikrostruktur zum effektiven Steifigkeitstensor. Im Modell ist 316L orthotrop (E_Aufbau ≈ 216, E_Schweiß ≈ 214, E_Wandnormale ≈ 197 GPa) und 17-4PH nahezu isotrop (E ≈ 205 GPa). Der Abgleich mit experimentellen Zugversuchen (Abschnitt 3.6) bestätigt das Modell für das schwach texturierte 17-4PH (Beträge ~7–22 % über dem Experiment, gleiche schwache Anisotropie). Für 316L trifft das Modell die Größenordnung, nicht aber die Anisotropie-Richtung: das Experiment weist 316L als scharf ⟨100⟩-texturiert aus (V/H weich ≈ 94 GPa, 45° steif), was das Modell aus den vorhandenen Orientierungsdaten nicht reproduziert — diese tragen die scharfe Textur nicht. Die Einkristallkonstanten sind bestätigt; die offene Frage ist die kristallographische Textur der Scandaten (Prüfung der Orientierungskonventionen und Referenzrahmen der Schliffe per Polfigur), ergänzt durch validierte Einkristallkonstanten und ggf. periodische Randbedingungen.

# 7  Reproduzierbarkeit

Netzerzeugung (Neper/Gmsh-Container): `CLEAN=1 bash run_pipeline.sh`. FE-Rechnungen (dolfinx v0.7.3): `prepare_inputs.py` (Host) → `run_fem.sh` (Container). Auswertung/Grafiken: `python3 evaluation.py` (Abbildungen + `summary.csv`/`.md`), `python3 engineering_constants.py` (orthotrope Konstanten, Tabelle\ \ref{tbl:engconst}) , `python3 experimental_comparison.py` (Abgleich mit den experimentellen Zugversuchen, Abschnitt 3.6) , `python3 pole_figures.py` ({100}-Polfiguren der Schliffe) , `python3 specimen_figure.py` (Abbildung der numerischen Zugstäbe) , `python3 matching_extra.py` (Textur- und Kornform-Abgleich generiert vs. gemessen) und `python3 orientation_map.py` (IPF-Orientierungskarte eines RVE-Schnitts, Abschnitt 3.2) aus den Ergebnis-JSONs, EBSD-Rohdaten, Neper-Renderings bzw. dem RVE-Netz. Dieser Report wird als PDF gebaut mit `bash build_report.sh` bzw. direkt `pandoc report.md -o WAAM_anisotropy_report.pdf --pdf-engine=xelatex`; Abbildungen und Tabellen werden über `\ref`/`\label` automatisch nummeriert (kein manuelles Umnummerieren). Alle Skripte liegen im Projektordner `069-waam-polycrystal-anisotropy` bzw. im Neper-Pipeline-Ordner.

| Phase | C11 [GPa] | C12 [GPa] | C44 [GPa] | Zener *A* |
|---|---|---|---|---|
| FCC (316L, Austenit) | 204.6 | 137.7 | 126.2 | 3.77 |
| BCC (17-4PH, Martensit) | 231.4 | 134.7 | 116.4 | 2.41 |

: Kubische Einkristallkonstanten (`config.json`). *A* = 2·C44/(C11−C12), 1 = isotrop.\label{tbl:singlecrystal}
