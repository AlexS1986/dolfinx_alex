---
title: "Steifigkeitsüberhöhung im Übergangsbereich einer WAAM-N=1-Probe (316L → 17-4PH)"
subtitle: "Kurzfassung — explizite 2D-Modellierung der gemessenen Mikrostruktur im ebenen Spannungszustand"
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
  \usepackage{enumitem}
  \setlist{topsep=2pt, itemsep=1pt, parsep=0pt}
  \setlength{\emergencystretch}{4em}
  \hyphenation{Auf-bau-rich-tung Über-gangs-be-reich Über-gangs-zo-ne Stei-fig-keits-ten-sor Mi-kro-struk-tur Kon-den-sa-tion Re-kon-struk-tion Ein-kris-tall-kon-stan-ten}
---

**Kernaussage.** Der Zugversuch weist im Grenzflächenbereich der hybriden WAAM-Probe N=1 einen lokalen E-Modul von 232,4 ± 13 GPa aus — mehr als in beiden Ausgangswerkstoffen (316L 162,7; 17-4PH 201,7 GPa). Die aus dem EBSD-Scan explizit modellierte Mikrostruktur reproduziert das nicht: der Übergangsbereich ist im Modell die *weichste* Zone. Entscheidend und simulationsunabhängig ist die Voigt-Schranke derselben Kornstruktur — die theoretische Obergrenze jeder denkbaren Anordnung dieser Körner mit diesen Konstanten. Sie liegt in allen sechs untersuchten Auswertefenstern 10 bis 50 % unter dem Messwert. Der Steifigkeitstensor des Übergangswerkstoffs selbst müsste also um den Faktor s ≈ 1,33 … 1,48 erhöht sein. Gegenprobe: über die volle Probenhöhe trifft dasselbe Modell den Monowerkstoff 316L auf 1,6 %.

*Diese Kurzfassung ist für sich lesbar. Herleitungen, Rechenbeispiele, Verifikationsprotokoll und die Quellenlage der Konstanten stehen im ausführlichen Bericht `WAAM_N1_transition_report.pdf`; Code und Ergebnisdateien im Projektordner `070-waam-n1-transition-2d`.*

# 1  Fragestellung

Die N=1-Probe kombiniert 316L und 17-4PH in einem Bauteil; dazwischen liegt ein Übergangsbereich unbekannter Zusammensetzung, in dem der Zugversuch (Lastachse ∥ Aufbaurichtung) eine erhöhte Steifigkeit ausweist.

| Messbereich | E [GPa] | ν | R_p0,2 [MPa] |
|---|---|---|---|
| Global | 212,4 ± 2 | 0,427 ± 0,048 | 342 ± 2 |
| 316L | 162,7 ± 2 | – | 283 ± 6 |
| 17-4PH | 201,7 ± 2 | – | 586 ± 11 |
| Grenzfläche | 232,4 ± 13 | – | 456 ± 20 |

: Experimentelle Kennwerte (extern gemessen), Orientierung V, Probendicke 2 mm.\label{tbl:exp}

![EBSD-Map des Übergangsbereichs (IPF-Z). Links 17-4PH — feiner Martensit in alten Austenitkörnern —, rechts 316L mit groben kolumnaren Chevron-Körnern, dazwischen der Übergangsbereich. Schwarzes Rechteck: der modellierte Ausschnitt (Referenzband); grünes Rechteck: die Übergangszone. Aufbaurichtung = −x, Lastachse = x.](fig_roi_map.png){#fig:roimap width=12.5cm}

Zu klären ist, ob die Überhöhung (1) durch die Mikrostruktur allein erklärbar ist — Kornform, Kornverteilung, Textur — oder ob (2) zusätzlich die Werkstoffkonstanten im Übergangsbereich erhöht sein müssen. Anders als in der Neper-Studie (069) mit statistisch äquivalenten RVEs geht es hier um einen räumlich konkreten Bereich; deshalb wird die gemessene Mikrostruktur direkt abgebildet.

# 2  Modell

**Mikrostruktur als Feld.** Da kein Rohdatenexport pro Messpunkt vorlag, wird die Kornkarte aus dem TSL-Kornexport (Schwerpunkt, mittlere Bunge-Winkel, Ellipsenfit, Fläche, Phase) und dem gerenderten IPF-Bild rekonstruiert: Jedes Pixel geht an das Korn, das eine Kostenfunktion aus Ellipsenabstand und Farbabweichung minimiert (mittlerer Farbfehler 16,7 von 255). Die Struktur wird nicht geometrisch vernetzt, sondern als Feld auf ein reguläres Netz gelegt: ein Q1-Element je EBSD-Pixel (3,371 µm). Korngrenzen verlaufen dadurch treppenförmig — für die effektive Steifigkeit unkritisch (< 0,5 % bei halbierter Auflösung), für lokale Spannungsspitzen relevant. Abb.\ \ref{fig:meshzoom} zeigt das Prinzip an einem Ausschnitt von 42 × 30 Elementen.

![Das reguläre Netz über der Kornstruktur, Ausschnitt aus der Übergangszone. (a) EBSD-Originalscan. (b) Modelleingang: ein Q1-Element je EBSD-Pixel; dünne Linien sind Elementkanten, dicke Linien die rekonstruierten Korngrenzen. Diese folgen dem starren Raster treppenförmig, statt geometrisch nachgebildet zu werden.](fig_mesh_zoom.png){#fig:meshzoom width=12.5cm}

**Kornweise Steifigkeit.** Jedes Korn erhält den kubischen Einkristalltensor seiner gemessenen Phase, gedreht mit seiner eigenen Orientierung über die 6 × 6-Bond-Rotation, $\mathbf{C}' = \mathbf{M}_\sigma\mathbf{C}\mathbf{M}_\sigma^{\mathsf T}$. Jedes Korn hat damit seinen eigenen Steifigkeitstensor, auch bei gleicher Phase und gleichem Werkstoff. (Die Transponierte ist keine Orthogonalitätsaussage — $\mathbf{M}_\sigma$ ist in Voigt-Notation nicht orthogonal —, sondern folgt aus der Invarianz der Formänderungsarbeit: $\mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\varepsilon = \mathbf{I}$, also $\mathbf{M}_\varepsilon^{-1} = \mathbf{M}_\sigma^{\mathsf T}$.)

| Phase | C11 | C12 | C44 | *A* | E⟨100⟩ | E⟨110⟩ | E⟨111⟩ |
|---|---|---|---|---|---|---|---|
| FCC (Austenit) | 204,6 | 137,7 | 126,2 | 3,77 | 93,8 | 193,5 | 299,8 |
| BCC (Martensit, Ferrit) | 231,4 | 134,7 | 116,4 | 2,41 | 132,3 | 220,4 | 283,3 |

: Kubische Einkristallkonstanten [GPa] und daraus folgende Richtungsmoduln; *A* = Zener-Anisotropie. Literaturwerte verwandter Werkstoffe (304-Typ-Austenit, reines α-Eisen bei Raumtemperatur), nicht für 316L/17-4PH gemessen — alle Absolutwerte skalieren mit ihnen.\label{tbl:sc}

![Richtungsmodul E_x jedes Korns aus seinem eigenen rotierten und kondensierten Tensor (Referenzband, s ≡ 1). (a) Karte, (b) flächengewichtete Verteilung je Zone. Die Spanne 95 … 300 GPa schöpft den Einkristall-Spielraum voll aus; die scharfe Linie bei 186 GPa ist ein einzelnes Korn, das die 316L-Zone dominiert.](fig_grain_Ex.png){#fig:grainEx width=12.5cm}

**Ebener Spannungszustand.** Es liegt nur ein Schliff vor. Der rotierte Tensor ist im Allgemeinen triklin, die üblichen Kurzformeln sind daher nicht anwendbar; stattdessen wird mit A = (xx, yy, xy) und B = (zz, yz, xz) exakt statisch kondensiert: $\underline\sigma_B = \mathbf{0} \Rightarrow \mathbf{C}^{\text{PS}} = \mathbf{C}_{AA} - \mathbf{C}_{AB}\mathbf{C}_{BB}^{-1}\mathbf{C}_{BA}$. Die Out-of-plane-Dehnungen relaxieren frei, die Lösung ist als dickengemittelt zu lesen. Der ebene Spannungszustand ist zugleich die *weichere* der beiden 2D-Idealisierungen — die Kernaussage wird dadurch konservativ geprüft.

**Werkstoff und Vorfaktor.** Die Zone bestimmt den Werkstoff (17-4PH / Übergang / 316L), die gemessene Phase des Korns das Kristallsystem. Der Übergangsbereich hat einen eigenen Konstantensatz, voreingestellt auf die Werte der Ausgangswerkstoffe. Ein skalarer Vorfaktor s(x) wirkt ausschließlich dort und skaliert alle Tensorkomponenten gleichmäßig — Betrag, nicht Anisotropierichtung:

$$ \mathbf{C}_{\text{Zelle}}(x) \;=\; s(x)\cdot\mathcal{P}\Big(\mathbf{M}(g_{\text{Korn}})\;\mathbf{C}^{\text{kub}}\big[\text{Werkstoff, Kristallsystem}\big]\;\mathbf{M}(g_{\text{Korn}})^{\mathsf T}\Big) $$

mit $\mathcal{P}$ der Kondensation aus dem vorigen Absatz; Voreinstellung ist s ≡ 1.

**Randwertproblem und Auswertung.** Lineare Elastostatik, verschiebungsgesteuerter Zug entlang x (ε₀ = 10⁻³), oben und unten spannungsfrei. Q1-Verschiebungen, Steifigkeit als DG0-Feld; Löser dolfinx v0.7.3 (CG + GAMG). Ausgewertet wird $E = \sum\sigma_{xx} / \sum\varepsilon_{xx}$ über eine Zone oder über Streifen von ≈ 100 µm quer zur Last — genau die experimentelle DIC-Vorschrift. Zusätzlich werden aus denselben kornweisen Tensoren Voigt- und Reuss-Schranke berechnet; sie sind solverunabhängig und tragen das zentrale Argument.

# 3  Ergebnisse

## 3.1  Die Mikrostruktur erklärt die Überhöhung nicht

Das Referenzfenster (2781 × 886 µm, 216 975 Zellen) löst 7601 Körner einzeln auf. Der Übergangsbereich ist darin die weichste Zone — das Vorzeichen des gemessenen Effekts kehrt sich um. Ursache ist der hohe FCC-Anteil (90,4 %) bei breiter Orientierungsstreuung: FCC-Austenit ist mit E⟨100⟩ = 94 GPa deutlich nachgiebiger als BCC-Martensit (132 GPa).

| Zone | Reuss | FE (dolfinx) | Voigt | Experiment |
|---|---|---|---|---|
| 17-4PH | 176,5 | 187,0 | 201,6 | 201,7 ± 2 |
| Übergang | 164,2 | 175,2 | 203,5 | 232,4 ± 13 |
| 316L | 185,8 | 186,0 | 186,0 | 162,7 ± 2 |

: Referenzband, E_x [GPa]. Die FE-Werte liegen in allen Zonen innerhalb ihrer Schranken; der gemessene Übergangswert liegt außerhalb.\label{tbl:ref}

Entscheidend ist die vorletzte Spalte: die Voigt-Schranke der Übergangszone liegt bei 203,5 GPa, 14 % unter dem Messwert. Keine noch so günstige Anordnung dieser Körner erreicht mit diesen Konstanten 232,4 GPa — diese Aussage hängt an keiner FE-Rechnung.

## 3.2  Höhenstudie: der Befund hält über die ganze Probenhöhe

Um auszuschließen, dass dies am gewählten Ausschnitt liegt, wurden vier weitere Bänder gleicher Höhe übereinander gelegt und zusätzlich ein Fenster über die volle Kartenhöhe gerechnet (905 025 Zellen, 30 673 Körner). Zonengrenzen sind nur für das Referenzband markiert; für die übrigen Fenster werden sie aus dem Gefüge bestimmt (Halbwertsstelle des BCC-Anteils, Beginn des grobkörnigen 316L). Nebenbefund: die Grenzfläche ist nicht ortsfest — sie wandert über die Probenhöhe um mehrere hundert µm, die Zonenbreite schwankt zwischen 199 und 1126 µm (Abb.\ \ref{fig:studymap}).

![(a) Die sechs Auswertefenster im EBSD-Scan: vier gestapelte Bänder (schwarz) und das Referenzband (weiß gestrichelt); grün die je Band bestimmten Zonengrenzen. Rechts außerhalb aller Fenster liegt homogenes 316L, das bewusst nicht mitmodelliert wird. (b) Lage und Breite der Übergangszone über die Probenhöhe.](fig_study_map.png){#fig:studymap width=15cm}

| Fenster | Körner | Reuss | FE | Voigt | Messwert / Voigt | s_erf |
|---|---|---|---|---|---|---|
| Band 1 (oben) | 12 928 | 131,5 | 142,9 | 166,5 | 1,40 | 1,63 |
| Band 2 | 9 174 | 159,0 | 169,4 | 198,8 | 1,17 | 1,37 |
| Band 3 | 5 266 | 170,0 | 181,5 | 210,8 | 1,10 | 1,28 |
| Band 4 (unten) | 3 132 | 134,5 | 140,1 | 154,9 | 1,50 | 1,66 |
| Referenzband | 7 601 | 164,2 | 175,2 | 203,5 | 1,14 | 1,33 |
| volle Höhe | 30 673 | 143,6 | 156,6 | 182,4 | 1,27 | 1,48 |

: Übergangszone in allen sechs Auswertefenstern, E_x [GPa]. In keinem erreicht die Voigt-Schranke den Messwert. s_erf = 232,4 / E_FE.\label{tbl:study}

Das Referenzband ist dabei kein günstig gewählter Ausschnitt, sondern hat von allen die zweithöchste Schranke; über die volle Probenhöhe wächst der Abstand auf 27 %.

**Ein echter Validierungspunkt.** Im Referenzband besteht die 316L-Zone zu 99,8 % aus *einem* Korn; ihr Modul war deshalb nicht mit dem gemessenen Aggregatwert vergleichbar. Über die volle Höhe enthält dieselbe Zone 388 Körner, und ihr Modul fällt von 186,0 auf 160,2 GPa — 1,6 % neben den gemessenen 162,7 GPa. Dasselbe Modell, dieselben Konstanten und dieselbe Idealisierung treffen also den Monowerkstoff auf 1,6 % und verfehlen den Übergangsbereich um 33 %. Der Fehler liegt sehr wahrscheinlich nicht im Modell, sondern in der Annahme, der Übergangswerkstoff habe dieselben Einkristallkonstanten wie seine Ausgangswerkstoffe. (Für 17-4PH bleibt eine Lücke: 172,4 gegenüber 201,7 GPa — dort sinkt der BCC-Anteil über die volle Höhe von 67 auf 48 %, und die α-Eisen-Konstanten enthalten die Ausscheidungshärtung nicht.)

![Voigt-/Reuss-Bereich (violett), arithmetischer Schätzer (offener Kreis) und FE-Lösung (Raute) je Auswertefenster gegen den Messwert (blau). (a) Übergangszone: der Messwert liegt in jedem Fenster oberhalb der Voigt-Schranke. (b) 316L-Zone: im Referenzband und in den Bändern 2 und 3 entartet der Schrankenbereich, weil dort ein einziges Korn dominiert; erst über die volle Höhe entsteht ein echter Aggregatwert — und der trifft den Messwert.](fig_study_bounds.png){#fig:studybounds width=14cm}

## 3.3  Lokaler Verlauf und erforderliche Erhöhung

Aus derselben Feldlösung folgt ein ortsaufgelöster Verlauf E(x). Der Zähler ⟨σ_xx⟩ ist über alle Streifen konstant (Kräftegleichgewicht, Variationskoeffizient 10⁻⁸ %) — die gesamte Struktur des Verlaufs steckt im Dehnungsfeld. Das Minimum liegt in jedem Fenster in oder am Rand der Übergangszone, nie in einem Monobereich.

Wegen der Serienschaltung gilt bei gleichmäßiger Skalierung $E_{\text{Zone}}(s) = s\cdot E_{\text{Zone}}(1)$, woraus der erforderliche Vorfaktor unmittelbar folgt: s = 232,4/175,2 = 1,33 (Referenzband) bzw. 232,4/156,6 = 1,48 (volle Höhe). Gerechnet wurden ein konstantes s = 1,33 und ein Gauß-Profil $s(x) = 1 + 0{,}5\exp[-((x-x_0)/\sigma)^2]$ mit σ = 350 µm. Beide treffen denselben Zonenmittelwert, erzeugen aber deutlich verschiedene räumliche Verläufe (Abb.\ \ref{fig:eprofile}) — das konstante s mit Sprüngen an den Zonengrenzen, das Gauß-Profil stetig, dafür mit höherer lokaler Spitze (≈ 273 statt ≈ 258 GPa). Aus dem Zonenmittelwert allein ist nicht zu entscheiden, welche Variante richtig ist; dafür bräuchte es einen ortsaufgelösten Messverlauf innerhalb der Grenzfläche mit bekannter Zuordnung zum EBSD-Ausschnitt. Genau deshalb ist s(x) als frei vorgebbare Funktion implementiert.

![(a) Lokaler E-Verlauf aus der FE-Lösung (28 Streifen) mit den experimentellen Zonenmittelwerten als Bänder, dazu die beiden Varianten mit überhöhter Übergangssteifigkeit. (b) Die angesetzten Vorfaktoren s(x). (c) Zerlegung des Quotienten für s ≡ 1: der Zähler ist über alle Streifen exakt konstant, die gesamte Variation stammt aus dem Nenner.](fig_E_profile.png){#fig:eprofile width=9.5cm}

**Was die Feldlösung zusätzlich leistet.** Für den Zonenmittelwert genügt bereits eine arithmetische Mittelung der kornweisen Moduln (über 18 Zonenwerte im Mittel 2,0 GPa neben der FE-Lösung); nicht ersetzbar ist sie für alles Lokale. σ_xx streut über den ROI von 64 bis 380 MPa, korreliert mit der Kornsteifigkeit aber nur mit r = 0,64 — rund 59 % der Spannungsstreuung stammen aus der Nachbarschaft, nicht aus dem Korn selbst; Spannungsspitzen erreichen das 2,1-fache des Mittelwerts.

**Eine messbare Spur für s(x).** Die Ferritscope-Messung stammt von derselben Probe und denselben 14 Messpunkten wie der lokale E-Verlauf. Das E-Maximum liegt bei Messpunkt 6, am Ende des steilen Ferritanstiegs von 7 auf 39 %; mit dem Ferritgehalt korreliert E nur schwach (r = 0,32), mit dem Betrag seines Gradienten besser (r = 0,43) — die Überhöhung sitzt dort, wo sich die Phasenzusammensetzung am schnellsten ändert.

# 4  Annahmen und Grenzen

- **Einkristallkonstanten sind Literaturwerte verwandter Werkstoffe** (304-Typ-Austenit, reines α-Eisen). Alle Absolutwerte skalieren mit ihnen; die Kernaussage hält, solange die tatsächlichen Konstanten des Übergangswerkstoffs nicht um mehr als ~14 % (Referenzband) bzw. ~27 % (volle Höhe) darüberliegen — und genau das wäre der zu belegende Effekt.
- **Unterschiedliche Auswertevolumina:** das experimentelle Fenster „Grenzfläche" ist mehrere Millimeter breit, die modellierte Übergangszone 0,2 … 1,1 mm. Dass der DIC-Verlauf an einem einzelnen Punkt auf 237 GPa steigt, spricht für eine real lokale Überhöhung.
- **Repräsentativität und Zweidimensionalität:** der EBSD-Scan stammt nicht aus derselben Probe wie der Zugversuch, die Zuordnung ROI ↔ DIC-Messpunkt ist unbekannt (die *innere* Repräsentativität ist mit der Höhenstudie geklärt). Bei 316L-Körnern über 1 mm Äquivalenzdurchmesser gegenüber 2 mm Probendicke ist die Schnittebene zudem kaum repräsentativ für die Dicke.
- **Kornkarte rekonstruiert, nicht gemessen;** je Korn wird die mittlere Orientierung angesetzt. Lineare Elastizität, ideale Kornbindung, keine Eigenspannungen.
- **Verifikation:** 17 Einheitentests der Kristallmathematik; Bond-Rotation gegen die exakte Tensorrotation vierter Stufe auf 2·10⁻¹³ GPa; Patch-Test exakt; alle 18 Zonenmoduln der Höhenstudie innerhalb ihrer Schranken; Kreuzvergleich mit einem unabhängigen numpy-Referenzsolver auf 1,3·10⁻¹¹ relativ; Netzstudie < 0,5 %.

# 5  Schlussfolgerungen

1. Die gemessene Steifigkeitsüberhöhung (232,4 ± 13 GPa) ist mit Kornstruktur und Kristallorientierung allein nicht erklärbar; das Modell macht den Übergang zur weichsten statt zur steifsten Zone.
2. Die Aussage ist stärker als ein Simulationsergebnis und hängt nicht am Ausschnitt: bereits die Voigt-Schranke liegt in allen sechs Auswertefenstern 10 bis 50 % unter dem Messwert (Referenzband 14 %, volle Probenhöhe 27 %).
3. Der Steifigkeitstensor des Übergangswerkstoffs müsste um den Faktor s ≈ 1,33 … 1,48 über dem der Ausgangswerkstoffe liegen — viel, denn zwischen Austenit und Martensit unterscheiden sich die Einkristallkonstanten von Stählen nur um etwa 10 %. Zu begründen wären Ausscheidungen, veränderte Phasenanteile oder ein Mischkristall.
4. Über die volle Probenhöhe trifft dasselbe Modell den Monowerkstoff 316L auf 1,6 % und verfehlt den Übergangsbereich um 33 %: die Diskrepanz liegt im Werkstoff, nicht im Modell.
5. Das Steifigkeitsmaximum fällt mit dem steilsten Gradienten des Ferritgehalts zusammen (dieselbe Probe). Damit ließe sich s(x) an eine Messgröße koppeln, statt es frei anzupassen.
6. Nächste Schritte: validierte Einkristallkonstanten; Gefügeanalyse in der Zone des steilsten Ferritgradienten; per-Messpunkt-EBSD-Export und ein Scan aus der geprüften Probe; Anpassung von s(x) an einen ortsaufgelösten Messverlauf.
