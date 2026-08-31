---
title: "Steifigkeitsüberhöhung im Übergangsbereich einer WAAM-N=1-Probe (316L → 17-4PH)"
subtitle: "Explizite 2D-Modellierung der gemessenen Mikrostruktur im ebenen Spannungszustand (dolfinx)"
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
  \hyphenation{Auf-bau-rich-tung Über-gangs-be-reich Über-gangs-zo-ne Stei-fig-keits-ten-sor Mi-kro-struk-tur Kon-den-sa-tion Re-kon-struk-tion Ein-kris-tall-kon-stan-ten}
---

# Zusammenfassung

Der Zugversuch an der hybriden WAAM-Probe N=1 (316L → 17-4PH, Lastachse ∥ Aufbaurichtung) zeigt im Grenzflächenbereich einen lokalen E-Modul von 232,4 ± 13 GPa — höher als in beiden Ausgangswerkstoffen (316L 162,7; 17-4PH 201,7 GPa). Untersucht wird, ob diese Überhöhung allein aus Kornstruktur und Kristallorientierung erklärbar ist.

**Kernergebnis: nein.** Mit der gemessenen Mikrostruktur allein ist der Übergangsbereich im Modell die weichste Zone (175,2 GPa gegenüber 187,0 für 17-4PH und 186,0 für 316L) — die Überhöhung wird nicht reproduziert, das Vorzeichen kehrt sich um. Stärker noch: die Voigt-Schranke der Übergangszone, also die theoretische Obergrenze jeder denkbaren Anordnung dieser Körner mit diesen Einkristallkonstanten, liegt bei 203,5 GPa und damit 14 % unter dem Messwert. Es muss zusätzlich der Steifigkeitstensor des Übergangswerkstoffs selbst erhöht sein; der dafür vorgesehene ortsabhängige Vorfaktor s(x) quantifiziert den erforderlichen Betrag zu s ≈ 1,33.

Dazu wurde die im EBSD-Scan gemessene Mikrostruktur eines 2,78 × 0,89 mm großen Ausschnitts explizit modelliert: kein statistisch äquivalentes Ersatzgefüge (wie in der Neper-Studie), sondern die tatsächliche Kornkarte, rekonstruiert aus dem TSL-Kornexport und dem IPF-Bild, auf ein reguläres Netz von 216 975 Zellen (1 Zelle = 1 EBSD-Pixel = 3,371 µm) abgebildet. Jede Zelle trägt den kubischen Einkristalltensor ihres Korns, rotiert mit dessen gemessener Orientierung und exakt auf den ebenen Spannungszustand kondensiert. 7601 Körner werden so einzeln aufgelöst.

**Der Befund hält über die ganze Probenhöhe.** Eine Höhenstudie (Kapitel 3) legt vier weitere Bänder gleicher Höhe übereinander und wertet zusätzlich ein Fenster über die volle Kartenhöhe aus (905 025 Zellen, 30 673 Körner). Alle sechs Fenster sind mit dolfinx gerechnet. In *keinem* erreicht die Voigt-Schranke der Übergangszone den Messwert; der Abstand beträgt 10 bis 50 %, über die volle Höhe 27 %, und der erforderliche Vorfaktor steigt dort auf s ≈ 1,48. Zugleich löst dieses große Fenster den Einkorn-Artefakt des Referenzbandes auf: die 316L-Zone enthält dort 388 statt 18 Körner, und ihr Modul fällt von 186,0 auf 160,2 GPa — 1,6 % neben dem gemessenen 316L-Wert von 162,7 GPa. Dasselbe Modell, das den Monowerkstoff auf 1,6 % trifft, verfehlt den Übergangsbereich um 33 %. Das ist der bislang stärkste Hinweis darauf, dass die Diskrepanz im Werkstoff und nicht im Modell liegt.

Ein Nebenergebnis betrifft den Mehrwert der Feldrechnung: Für den Zonenmittelwert genügt bereits eine einfache Mittelung der kornweisen Moduln. Was nur die Feldlösung liefert, ist das lokale Spannungsfeld — Zellen mit identischer Kornsteifigkeit tragen je nach Nachbarschaft Spannungen zwischen 127 und 328 MPa, und nur rund 41 % der Spannungsstreuung erklären sich aus der Steifigkeit des Korns selbst.

*Aufbau:* Kapitel 2 enthält die Ergebnisse des Referenzbandes geschlossen und ohne Vorgriff auf die Methodik, Kapitel 3 die Höhenstudie, Kapitel 4–6 Diskussion, Schlussfolgerungen und Grenzen. Der ausführliche Modell- und Methodenteil folgt in Kapitel 7; Herleitungen, Rechenbeispiele und die Verifikation stehen in den Anhängen A–D.

# 1  Zielsetzung

Die N=1-Probe kombiniert 316L und 17-4PH in einem Bauteil; zwischen den Monobereichen liegt ein Übergangsbereich unbekannter Zusammensetzung. Die lokale DIC-Auswertung des Zugversuchs weist dort eine erhöhte Steifigkeit aus (Tabelle\ \ref{tbl:exp}). Ziel ist zu prüfen, ob diese Überhöhung

1. durch die Mikrostruktur (Kornform, Kornverteilung, kristallographische Textur) erklärbar ist, oder ob
2. zusätzlich die Einkristall-/Werkstoffkonstanten im Übergangsbereich erhöht sein müssen — was materialkundlich zu begründen wäre (Ausscheidungen, Mischkristall, veränderte Phasenanteile).

| Messbereich | E [GPa] | ν | G [GPa] | R_p0,2 [MPa] | R_m [MPa] | A [%] |
|---|---|---|---|---|---|---|
| Global | 212,4 ± 2 | 0,427 ± 0,048 | 74,5 ± 2 | 342 ± 2 | 606 ± 4 | 24,06 ± 3,50 |
| 316L | 162,7 ± 2 | – | – | 283 ± 6 | 606 ± 4 | 103,41 ± 8,63 |
| 17-4PH | 201,7 ± 2 | – | – | 586 ± 11 | – | – |
| Grenzfläche | 232,4 ± 13 | – | – | 456 ± 20 | – | – |

: Experimentelle Kennwerte des Zugversuchs (extern gemessen), Orientierung V, Probendicke 2 mm. An diesen Werten muss sich das Modell messen lassen.\label{tbl:exp}

Abgrenzung zur Neper-Studie (Projekt 069): dort wurden aus EBSD-Statistiken *statistisch äquivalente* 3D-RVEs erzeugt. Hier geht es um einen einzelnen, räumlich konkreten Bereich mit einem materialkundlichen Gradienten — ein statistisches Ersatzgefüge wäre dafür ungeeignet, weil gerade die reale örtliche Anordnung interessiert. Deshalb wird die gemessene Mikrostruktur direkt abgebildet.

*Sprachregelung:* „Übergangsbereich" meint den physikalischen Bereich der Probe zwischen den beiden Monowerkstoffen, „Übergangszone" dessen Abbildung im Modell (Zone 1, x = 666,6 … 1494,5 µm). Im Zugversuchsbericht heißt derselbe Bereich „Grenzfläche".

# 2  Ergebnisse

## 2.1  Das Modell in Kürze

Tabelle\ \ref{tbl:modell} fasst zusammen, was gerechnet wurde, so weit es zum Verständnis der Ergebnisse nötig ist; die vollständige Beschreibung steht in Kapitel 7.

| Aspekt | Umsetzung |
|---|---|
| Gefüge | explizit aus dem EBSD-Scan übernommen, 7601 Körner einzeln aufgelöst — kein statistisches Ersatzgefüge |
| Ausschnitt (ROI) | 2781 × 886 µm, Lastachse = x = Aufbaurichtung; Zonen 17-4PH / Übergang / 316L (Abb.\ \ref{fig:roimap}) |
| Netz | reguläres Q1-Netz, 1 Element = 1 EBSD-Pixel = 3,371 µm, 216 975 Elemente; Korngrenzen als Feld, nicht geometrisch vernetzt |
| Steifigkeit je Zelle | kubischer Einkristalltensor der gemessenen Phase, mit der gemessenen Kornorientierung rotiert und exakt auf den ebenen Spannungszustand kondensiert |
| Werkstoffzuordnung | Zone bestimmt den Werkstoff (17-4PH / Übergang / 316L), die gemessene Phase des Korns das Kristallsystem (BCC/FCC) |
| Stellschraube | skalarer Vorfaktor s(x) auf den Tensor der Übergangszone, Voreinstellung s ≡ 1 |
| Belastung | verschiebungsgesteuerter Zug in x, ε₀ = 10⁻³; oben und unten spannungsfrei (Querkontraktion unbehindert) |
| Auswertung | E = Σσ_xx / Σε_xx über eine Zone bzw. über Streifen von 99 µm Breite — dieselbe Vorschrift wie bei der DIC-Auswertung |
| Löser | dolfinx v0.7.3; alle berichteten Zahlen stammen aus dieser Rechnung |

: Das Modell auf einen Blick. Details: Kapitel 7; Herleitung der Tensorrotation: Anhang A; Entstehung eines einzelnen E(x)-Datenpunkts: Anhang B.\label{tbl:modell}

![EBSD-Map des Übergangsbereichs. Links 17-4PH (feiner Martensit in alten Austenitkörnern), rechts 316L (grobe kolumnare Chevron-Körner), dazwischen der Übergangsbereich. Schwarzes Rechteck: modellierter ROI; grünes Rechteck: Übergangszone. Aufbaurichtung = −x, Lastachse = x.](fig_roi_map.png){#fig:roimap width=14cm}

**Einkristallkonstanten.** Alle Absolutwerte des Modells skalieren mit Tabelle\ \ref{tbl:singlecrystal}. Es handelt sich um Literaturwerte verwandter Werkstoffe (304-Typ-Austenit bzw. reines α-Eisen), nicht um gemessene Werte für 316L, 17-4PH oder den Übergangswerkstoff; Herkunft und Belastbarkeit stehen in Anhang E.

| Phase | C11 [GPa] | C12 [GPa] | C44 [GPa] | Zener *A* | E⟨100⟩ | E⟨110⟩ | E⟨111⟩ | Herkunft |
|---|---|---|---|---|---|---|---|---|
| FCC (Austenit, 316L) | 204,6 | 137,7 | 126,2 | 3,77 | 93,8 | 193,5 | 299,8 | Cr-Ni-Stahl 304-Typ [1] |
| BCC (Martensit, 17-4PH) | 231,4 | 134,7 | 116,4 | 2,41 | 132,3 | 220,4 | 283,3 | α-Eisen bei RT [2–4] |

: Kubische Einkristallkonstanten (`config.json`) und die daraus folgenden Richtungsmoduln [GPa]. FCC gilt auch für den Rest-Austenit im 17-4PH, BCC auch für Ferrit im 316L. *A* = 2·C44/(C11−C12), 1 = isotrop. Der Übergangsbereich verwendet dieselben Werte als Voreinstellung.\label{tbl:singlecrystal}

**Aufgelöste Mikrostruktur.** Im ROI werden 7601 Körner einzeln aufgelöst — jedes mit eigener Orientierung und damit eigenem Steifigkeitstensor:

| Zone | Körner | Fläche des größten Korns | Flächengew. Median d_eq |
|---|---|---|---|
| 17-4PH | 5771 | 14,1 % der Zone | 26,9 µm |
| Übergang | 1863 | 24,4 % der Zone | 122,9 µm |
| 316L | 18 | 99,8 % der Zone | 1219 µm |

: Kornstatistik im modellierten Ausschnitt. Die 316L-Zone besteht praktisch aus einem einzigen Korn.\label{tbl:grains}

Der letzte Eintrag ist für die Interpretation zentral: der 316L-Bereich des ROI ist kein Polykristall, sondern zu 99,8 % ein einziges kolumnares Chevron-Korn; über den gesamten ROI nimmt dieses eine Korn 54,7 % der Fläche ein. Sein „Zonenmodul" ist damit der Richtungsmodul einer einzelnen Kristallorientierung, kein Aggregatmittelwert — er ist nicht mit dem experimentellen 316L-Wert vergleichbar, der über viele Millimeter und damit über viele Körner mittelt (Kapitel 4).

**Kornweise Steifigkeit.** Abb.\ \ref{fig:grainEx} zeigt den Richtungsmodul E_x jedes Korns, berechnet aus dessen eigenem rotierten und kondensierten Tensor, für den Grundfall s(x) = 1. Die Spanne über den ROI beträgt 95 … 300 GPa und schöpft damit den Einkristall-Spielraum voll aus (Tabelle\ \ref{tbl:singlecrystal}: E⟨100⟩ = 94 GPa bis E⟨111⟩ = 300 GPa für FCC). Die Verteilung ist in den beiden feinkörnigen Zonen breit; im 316L-Bereich degeneriert sie zu einer einzelnen Linie bei 186 GPa — die grafische Bestätigung des Einkorn-Befunds aus Tabelle\ \ref{tbl:grains}.

![Richtungsmodul E_x je Korn aus dessen eigenem rotierten, plane-stress-kondensierten Tensor, für den Fall s(x) = 1 (reine Mikrostruktur, ohne Skalierung). (a) Karte, (b) flächengewichtete Verteilung je Zone; die scharfe Linie bei 186 GPa ist das eine dominante 316L-Korn. Dasselbe Feld für die beiden skalierten Varianten zeigt Abb.\ \ref{fig:grainExvar}.](fig_grain_Ex.png){#fig:grainEx width=16cm}

## 2.2  Kernbefund: die Mikrostruktur erklärt die Überhöhung nicht

**Analytische Schranken — ohne jede FE-Rechnung.** Der erreichbare Bereich lässt sich allein aus den kornweisen Tensoren eingrenzen. Die Voigt-Schranke (einheitliche Dehnung) und die Reuss-Schranke (einheitliche Spannung) folgen aus den flächengewichteten Mitteln

$$ \mathbf{C}^{\text{Voigt}} = \langle \mathbf{C}^{\text{PS}}\rangle, \qquad
\mathbf{S}^{\text{Reuss}} = \langle (\mathbf{C}^{\text{PS}})^{-1}\rangle,
\qquad E_x = 1/S_{11}. $$

| Zone | Reuss (untere Schranke) | Voigt (obere Schranke) | FE (dolfinx, s = 1) | Experiment (DIC) |
|---|---|---|---|---|
| 17-4PH | 176,5 | 201,6 | 187,0 | 201,7 ± 2 |
| Übergang | 164,2 | 203,5 | 175,2 | 232,4 ± 13 |
| 316L | 185,8 | 186,0 | 186,0 | 162,7 ± 2 |
| ROI gesamt | 176,8 | 207,6 | 182,9 | 212,4 ± 2 (ganze Probe) |

: Voigt-/Reuss-Schranken der rekonstruierten Kornstruktur, FE-Ergebnis und Messwerte, alle E_x in GPa. Die FE-Werte liegen in allen Zonen innerhalb ihrer Schranken (Konsistenzprüfung). Der gemessene Übergangswert liegt außerhalb.\label{tbl:bounds}

Dies ist der zentrale Befund, und er ist solverunabhängig: Die Voigt-Schranke ist die theoretische Obergrenze über alle denkbaren Anordnungen der vorhandenen Körner mit den gegebenen Einkristallkonstanten. Der gemessene Wert 232,4 GPa liegt 14 % über dieser Obergrenze (203,5 GPa). Keine noch so günstige Kornanordnung kann ihn mit diesen Konstanten erreichen.

Auffällig ist zudem die schmale Schranke im 316L-Bereich (185,8 … 186,0 GPa): bei nur einem Korn fallen Voigt- und Reuss-Schranke zusammen, weil kein Steifigkeitskontrast vorliegt.

![Voigt-/Reuss-Bereich der rekonstruierten Kornstruktur je Zone (violett), FE-Ergebnis (rot) und Messwert (blau). Der gemessene Übergangswert liegt oberhalb der Voigt-Schranke.](fig_bounds.png){#fig:bounds width=12cm}

**Das FE-Ergebnis mit der gemessenen Mikrostruktur (s ≡ 1).**

| Größe | Wert |
|---|---|
| E_apparent (ROI gesamt) | 182,87 GPa |
| ν_xy (scheinbar) | 0,0582 |
| E_lokal 17-4PH | 187,04 GPa |
| E_lokal Übergang | 175,19 GPa |
| E_lokal 316L | 185,96 GPa |
| ⟨σ_xx⟩ (alle Zonen) | 0,182865 GPa |

: FE-Ergebnis für die gemessene Mikrostruktur ohne Skalierung, ε₀ = 10⁻³ (`E_roi.json`).\label{tbl:fe}

Der Übergangsbereich ist im Modell die weichste der drei Zonen — das Vorzeichen des gemessenen Effekts kehrt sich um. Ursächlich ist der hohe FCC-Anteil (90,4 % der Zonenfläche) bei gleichzeitig breiter Orientierungsstreuung: FCC-Austenit ist mit E⟨100⟩ = 94 GPa in der weichen Richtung deutlich nachgiebiger als BCC-Martensit (132 GPa), und die Reuss-Schranke der Zone (164 GPa) ist entsprechend die niedrigste aller drei Zonen.

Die identische mittlere Spannung in allen drei Zonen (letzte Zeile) bestätigt das Gleichgewicht in Lastrichtung und ist zugleich der laufende Konsistenztest jeder Rechnung. Der auffällig niedrige scheinbare Querkontraktionskoeffizient ν_xy = 0,058 ist kein Fehler, sondern eine Folge des einen dominanten 316L-Korns; er wird in Kapitel 4 erklärt.

## 2.3  Erforderliche Erhöhung des Steifigkeitstensors: s ≈ 1,33

Wegen der Serienschaltung (σ_xx praktisch konstant entlang x) gilt für die Zonensteifigkeit bei gleichmäßiger Skalierung des Zonentensors in sehr guter Näherung

$$ E_{\text{Zone}}(s) \;=\; s\cdot E_{\text{Zone}}(s{=}1). $$

Daraus folgt der erforderliche Vorfaktor direkt aus Tabelle\ \ref{tbl:fe}:

$$ s_{\text{erf}} \;=\; \frac{232{,}4}{175{,}19} \;=\; 1{,}327 \;\approx\; 1{,}33 . $$

Der Steifigkeitstensor des Übergangswerkstoffs müsste also um ein Drittel über dem der Ausgangswerkstoffe liegen. Als glatte Alternative zum Sprung an den Zonengrenzen ist ein Gauß-Profil vorgesehen,

$$ s(x) \;=\; 1 + A\,\exp\!\left[-\left(\frac{x-x_0}{\sigma}\right)^{2}\right],
\qquad x_0 = 1050\ \text{µm},\ \ \sigma = 350\ \text{µm},\ \ A = 0{,}50, $$

mit x₀ in der Mitte der Übergangszone; die Amplitude ist so gewählt, dass der Zonenmittelwert denselben Messwert trifft. Physikalisch ist das die plausiblere Variante: entsteht die Versteifung durch ein Misch-/Diffusionsprofil (Cu-, Ni-, Cr-Gradient, Ausscheidungen), ist sie in der Mitte der Mischzone am stärksten und klingt zu den Monobereichen hin ab, statt an einer geometrischen Grenze zu springen. Beide Varianten sind über `run_fem.sh` als Standardfälle hinterlegt.

## 2.4  Lokaler E-Verlauf und die skalierten Varianten

Dieselbe Rechnung liefert neben den Zonenwerten einen ortsaufgelösten Verlauf E(x): Der ROI wird in 28 Streifen zu je 99,32 µm quer zur Lastachse zerlegt, und für jeden Streifen wird das Verhältnis der Summen aus Spannung und Dehnung über seine rund 7600 Elemente gebildet — dieselbe Vorschrift, die auch die DIC-Auswertung anwendet. Es gibt keine separate Rechnung je Streifen; alle Datenpunkte stammen aus derselben einen Feldlösung. Anhang B rechnet einen einzelnen Datenpunkt vollständig vor.

**Woher die Variation kommt.** Die Zerlegung in Zähler und Nenner (Abb.\ \ref{fig:eprofile}c) zeigt den Mechanismus unmittelbar:

- Der Zähler ⟨σ_xx⟩ ist über alle 28 Streifen konstant — der Variationskoeffizient liegt bei 10⁻⁸ %, ist also numerisch null. Das ist kein Zufall, sondern das Kräftegleichgewicht: die Streifen liegen in Reihe hintereinander, die durch jeden Querschnitt übertragene Kraft ist dieselbe.
- Der Nenner ⟨ε_xx⟩ variiert mit einem Variationskoeffizienten von 4,7 %.
- Folglich ist jede Struktur im E(x)-Verlauf ausschließlich eine Struktur des lokalen Dehnungsfeldes: weiche Streifen dehnen sich stärker, der Quotient sinkt. Der Verlauf reicht von 159,9 GPa (weichster Streifen, in der Übergangszone) bis 194,8 GPa.

Wichtig für die Interpretation der Felder in Abb.\ \ref{fig:fefields}: innerhalb eines Streifens streut σ_xx durchaus stark von Korn zu Korn (Standardabweichung rund 13 % des Mittelwerts) — steife Körner ziehen Last an, weiche geben sie ab. Diese Streuung mittelt sich über den Querschnitt jedoch exakt weg, sodass der Streifenmittelwert konstant bleibt.

![FE-Felder aus der dolfinx-Lösung. (a) Spannung in Lastrichtung: die korngenaue Streuung ist erheblich, doch der Mittelwert über einen Querschnitt ist überall gleich. (b) Dehnung in Lastrichtung: hier liegt die gesamte Ortsabhängigkeit von E(x); der Übergangsbereich dehnt sich am stärksten. Rot: der in Anhang B ausgewertete Streifen; weiß gestrichelt: Zonengrenzen.](fig_fe_fields.png){#fig:fefields width=15.5cm}

![(a) Lokaler E-Verlauf aus der FE-Lösung (28 Streifen zu je 99 µm) mit den experimentellen Zonenmittelwerten als Bänder über dem x-Bereich der jeweiligen Modellzone; rot markiert der Beispielstreifen aus Anhang B. Zusätzlich die beiden Varianten mit überhöhter Übergangssteifigkeit — durchgezogen, wenn ein dolfinx-Lauf vorliegt, gestrichelt, wenn aus der s = 1-Lösung abgeleitet. (b) Die angesetzten Vorfaktoren s(x); sie wirken ausschließlich in der Übergangszone. (c) Zerlegung des Quotienten für s(x) = 1: der Zähler ist über alle Streifen exakt konstant (Gleichgewicht), die gesamte Variation von E(x) stammt aus dem Nenner.](fig_E_profile.png){#fig:eprofile width=13.5cm}

**Was die überhöhten Lösungen bewirken.** Zu unterscheiden sind zwei Wirkungen:

- **Konstanter Faktor s = 1,33.** Der E-Verlauf der Übergangszone wird als Ganzes um 33 % angehoben, seine *Form* bleibt unverändert — das lokale Minimum bei x ≈ 0,85 mm bleibt ein Minimum, nur auf höherem Niveau. An beiden Zonengrenzen entsteht ein Sprung. Der Zonenmittelwert trifft den Messwert, der Verlauf innerhalb der Zone aber nicht: er ist am Rand zu hoch und in der Mitte zu niedrig.
- **Gauß-Profil.** Die Anhebung ist ortsabhängig und in der Zonenmitte am stärksten. Der Übergang zu den Monobereichen erfolgt stetig, es entsteht kein Sprung. Der Preis ist eine höhere lokale Spitze (≈ 273 GPa gegenüber ≈ 258 GPa bei konstantem s), weil dieselbe mittlere Anhebung auf einen schmaleren Bereich konzentriert wird.

Beide Varianten erfüllen also dieselbe Randbedingung — den gemessenen Zonenmittelwert — mit deutlich verschiedenen räumlichen Verläufen. Welche der beiden richtig ist, lässt sich aus dem Zonenmittelwert allein nicht entscheiden. Dafür bräuchte es einen ortsaufgelösten Messverlauf innerhalb der Grenzfläche, dessen Zuordnung zum EBSD-Ausschnitt bekannt ist. Das ist der Grund, warum s(x) als frei vorgebbare Funktion und nicht als Konstante implementiert wurde: sobald ein solcher Verlauf vorliegt, ist s(x) direkt daran anpassbar.

Auf die kornweisen Moduln wirkt s(x) nur im Betrag, nicht in der Richtungsabhängigkeit (Abb.\ \ref{fig:grainExvar}): Das Kornmuster bleibt identisch, die Übergangszone wird lediglich als Ganzes heller (konstantes s) beziehungsweise mit einem Maximum in der Zonenmitte (Gauß-Profil). Aufschlussreich sind die Histogramme: Beim konstanten Faktor verschiebt sich die gesamte Verteilung starr um 33 %, ihre Form bleibt erhalten. Beim Gauß-Profil wird sie deutlich breiter, weil Körner gleicher Orientierung je nach Position unterschiedlich stark angehoben werden. Beide Varianten erreichen denselben Zonenmittelwert, erzeugen aber verschiedene lokale Steifigkeits- und damit Spannungsfelder.

![Kornweiser Richtungsmodul E_x für die drei Modellvarianten, gemeinsame Farbskala. Links die Karten, rechts die flächengewichtete Verteilung innerhalb der Übergangszone (gestrichelt: Mittelwert). Das Kornmuster bleibt in allen Fällen identisch — s(x) skaliert nur den Betrag.](fig_grain_Ex_variants.png){#fig:grainExvar width=16cm}

**Zum Status der beiden Kurven in Abb.\ \ref{fig:eprofile}a.** Liegen dolfinx-Läufe der Varianten vor, zeigt die Abbildung deren Profile (durchgezogen). Andernfalls werden sie aus der s = 1-Lösung abgeleitet (gestrichelt) und als solche beschriftet. Die Ableitung nutzt die exakte Gleichgewichtsaussage von oben: da die mittlere Spannung jedes Querschnitts durch das Kräftegleichgewicht festliegt, hebt eine Skalierung der Zonensteifigkeit den Streifenmodul im selben Verhältnis an, $E_{s}(x_k) = s(x_k)\cdot E_{1}(x_k)$. Vernachlässigt wird dabei ausschließlich die Umverteilung der Dehnung *innerhalb* eines Streifens und in den schmalen Randbereichen an den Zonengrenzen; die Näherung ist daher gut, aber nicht exakt (Konsistenzprüfung in Anhang B).

**Verhältnis zum gemessenen DIC-Verlauf.** Die Bänder in Abb.\ \ref{fig:eprofile}a stammen aus den Auswertefenstern des Zugversuchsberichts (Tabelle\ \ref{tbl:exp}), nicht aus dem 14-Punkt-DIC-Verlauf. Letzterer erfasst die gesamte Messstrecke und ist daher separat in Abb.\ \ref{fig:dicprofile} dargestellt — er ist kein Ausschnitt aus Abb.\ \ref{fig:eprofile} und umgekehrt: es sind zwei verschiedene Auswertungen mit verschiedenen Abszissen (Ort im EBSD-ROI gegenüber Messpunktindex über die ganze Probe). Die Lage des simulierten Fensters innerhalb der Messstrecke ist nicht bekannt; die Markierung in Abb.\ \ref{fig:dicprofile} ist eine Annahme.

![Experimenteller lokaler E-Verlauf über die gesamte Messstrecke (DIC, 14 Messpunkte). Grau: Breite des im Modell erfassten Fensters (≈ 2,8 mm) an angenommener Position. Der Peak von 237 GPa an einem einzelnen Messpunkt zeigt, dass die Überhöhung räumlich eng begrenzt ist.](fig_dic_profile.png){#fig:dicprofile width=12cm}

## 2.5  Was die Feldlösung leistet — und was schon eine Mittelung liefert

Naheliegende Frage: Wenn man die kornweisen Moduln aus Abb.\ \ref{fig:grainEx} über dieselben Streifen mittelt — kommt dann dasselbe heraus wie aus der FE-Rechnung?

| Mittelung | mittlere Abweichung zur FE-Lösung | Korrelation mit E_FE(x) |
|---|---|---|
| arithmetisch ⟨E_x⟩ | +0,0 GPa (max. 9,8 GPa) | 0,957 |
| harmonisch = Tensor-Reuss | +5,7 GPa | 0,878 |
| Tensor-Voigt | −9,1 GPa | 0,473 |

: Streifenweiser Vergleich der Mittelungen mit der FE-Lösung (28 Streifen, s(x) = 1). Positive Werte bedeuten: die FE-Lösung ist steifer als die Mittelung. Herleitung und Einordnung: Anhang C.\label{tbl:averaging}

**Nein, identisch sind sie nicht — aber die arithmetische Mittelung kommt bemerkenswert nahe.** Über alle 28 Streifen verschwindet ihre mittlere Abweichung praktisch (+0,0 GPa), die Korrelation beträgt 0,957, die größte Einzelabweichung liegt bei rund 5 %. Für die Kernaussage ändert die Mittelung nichts: auch sie liefert im Übergangsbereich einen Zonenwert von 174,4 GPa gegenüber gemessenen 232,4 GPa. Für eine schnelle Abschätzung des E-Verlaufs genügt sie also — sie braucht keine FE-Rechnung.

![Streifenweise Mittelung der kornweisen Moduln aus Abb.\ \ref{fig:grainEx} im Vergleich zur FE-Lösung. (a) Alle vier Kurven über denselben 28 Streifen. (b) Differenz zur FE-Lösung: die arithmetische Mittelung liegt durchgehend innerhalb weniger GPa, die harmonische ist zu weich, die Tensor-Voigt-Schranke zu steif. Im einkristallinen 316L-Bereich (rechts) fallen alle Kurven exakt zusammen.](fig_averaging.png){#fig:averaging width=14cm}

**Was dabei verlorengeht.** Eine Mittelung liefert je Streifen genau eine Zahl; die FE-Rechnung liefert das vollständige Feld — und dieses Feld enthält die Wechselwirkung zwischen benachbarten Körnern, die in keiner Mittelungsvorschrift steckt. Konkret, aus derselben Lösung (Abb.\ \ref{fig:fefields}a):

- **Lastumlagerung von weichen auf steife Nachbarn.** Die Spannung in Lastrichtung streut über den ROI von 64 bis 380 MPa um einen Mittelwert von 183 MPa (Standardabweichung 12 %, 1./99. Perzentil 120/249 MPa). Steife Körner ziehen Last an, weiche geben sie ab — die Korrelation zwischen der Kornsteifigkeit E_x und der tatsächlich getragenen Spannung beträgt r = 0,64. Dass sie deutlich unter 1 liegt, ist der entscheidende Punkt: nur rund 41 % der Spannungsstreuung erklärt sich aus der Steifigkeit des Korns selbst, der Rest stammt aus seiner Umgebung.
- **Der Umgebungseinfluss direkt gemessen.** Betrachtet man ausschließlich Zellen mit praktisch identischer Kornsteifigkeit (E_x = 200 … 205 GPa, 3216 Zellen), so tragen diese trotzdem Spannungen von 127 bis 328 MPa (192 ± 25 MPa). Gleiche Orientierung, gleiches Material, gleiche Phase — und dennoch ein Faktor 2,6 zwischen der am geringsten und der am stärksten belasteten Zelle. Der Unterschied kommt allein aus der Nachbarschaft. Keine Mittelung, gleich welcher Art, kann das abbilden.
- **Spannungsüberhöhung.** Die höchste Zellspannung erreicht das 2,1-fache des Mittelwerts. Solche Überhöhungen treten an Korngrenzen, Tripelpunkten und an der Grenze zwischen fein- und grobkörnigem Gefüge auf und sind der Ansatzpunkt für Ermüdung und Rissinitiierung — Größen, für die der Mittelwert per Konstruktion blind ist.
- **Querdehnungsbehinderung.** Die Nachbarkörner behindern sich gegenseitig in der Querkontraktion. Genau das ist der Grund, warum weder die Voigt- noch die Reuss-Annahme die Zellspannungen trifft: Unterstellt man einheitliche Dehnung, liegt die vorhergesagte Zellspannung im Mittel um 8,7 % daneben (örtlich bis über 90 %); unterstellt man einheitliche Spannung, um 7,8 %.

**Fazit für die Verwendung:** Für den makroskopischen Kennwert — und damit für die Kernaussage dieses Berichts — genügt die Mittelung. Für alles, was lokal ist (Spannungsspitzen, Lastpfade, Grenzflächeneffekte, Ausgangsgrößen für Ermüdungs- oder Schädigungsmodelle), ist die Feldlösung nicht ersetzbar. Der Mehrwert der expliziten Mikrostrukturmodellierung liegt daher nicht primär im Zonenmittelwert, sondern im lokalen Feld.

## 2.6  Eine messbare Spur für s(x): der Ferritgehalt

Der Vorfaktor s(x) ist bisher eine reine Anpassungsgröße. Es gibt jedoch einen unabhängigen Messdatensatz, der ihm eine physikalische Bedeutung geben könnte: die Ferritscope-Messung der gradierten Probe. Sie stammt von derselben Probe (MHo1030_A9D_5) und denselben 14 Messpunkten wie der lokale E-Verlauf — der Vergleich braucht also keine Registrierungsannahme (Abb.\ \ref{fig:ferrite}).

Der Ferritgehalt springt zwischen den Messpunkten 4 und 6 von 7 % auf 39 % und sättigt danach bei etwa 45 %. Das Maximum des E-Moduls liegt bei Messpunkt 6, unmittelbar am Ende dieses steilen Anstiegs; der stärkste Gradient des Ferritgehalts liegt bei Punkt 5. Die Korrelation mit dem Ferritgehalt selbst ist schwach (r = 0,32), weil der Ferritgehalt sättigt, während E wieder abfällt — mit dem Betrag des Gradienten korreliert E besser (r = 0,43). Anders gesagt: Die Steifigkeitsüberhöhung sitzt nicht dort, wo viel Ferrit ist, sondern dort, wo sich die Phasenzusammensetzung am schnellsten ändert.

Das ist genau die Signatur, die man von einem Erstarrungs- oder Aufmischungseffekt erwarten würde: In der Mischzone ändern sich Zusammensetzung, Erstarrungsmodus und Abkühlbedingungen; dort könnten Gefügebestandteile entstehen, die in keinem der beiden Monowerkstoffe vorkommen.

![Ferritgehalt und lokaler E-Modul entlang der Messstrecke, beide an derselben Probe und denselben 14 Messpunkten gemessen. (a) Lokaler E-Modul aus der DIC. (b) Ferritgehalt (Ferritscope, Mittel aus drei Messreihen) und Betrag seines Gradienten. Grau: Bereich des steilsten Anstiegs. Das Steifigkeitsmaximum liegt am Ende des Übergangs, nicht im ferritreichen Monobereich.](fig_ferrite.png){#fig:ferrite width=12.5cm}

# 3  Höhenstudie: mehrere Bänder und die volle Probenhöhe

## 3.1  Fragestellung und Auswertefenster

Kapitel 2 wertet ein einzelnes horizontales Band in der Mitte der EBSD-Karte aus — dasjenige, für das eine Markierung vorliegt. Daraus ergeben sich zwei Fragen, die dieses Kapitel beantwortet:

1. **Ist dieses Band repräsentativ?** Der Kernbefund lautet, dass der Messwert oberhalb der Voigt-Schranke liegt. Träfe das nur für ein zufällig gewähltes Band zu, wäre die Aussage schwach.
2. **Lässt sich der Einkorn-Artefakt des 316L-Bereichs beheben?** Im Referenzband besteht die 316L-Zone zu 99,8 % aus einem einzigen Korn (Tabelle\ \ref{tbl:grains}) und ist deshalb kein Aggregat — der Vergleich mit dem gemessenen 316L-Wert war nicht zulässig.

Dazu werden vier weitere Bänder gleicher Höhe (886 µm) direkt übereinander über die Karte gelegt sowie ein Fenster über die volle Kartenhöhe. Horizontal bleiben alle Fenster auf x = 30 … 2811 µm beschränkt wie das Referenzband; der weit rechts liegende, homogene 316L-Bereich wird also bewusst nur teilweise erfasst (Abb.\ \ref{fig:studymap}a). Tabelle\ \ref{tbl:studycases} listet sie auf. Alle Fenster stammen aus demselben Scan und derselben Pixel→Korn-Rekonstruktion; die Rekonstruktion läuft ohnehin über die ganze Karte und wird nur einmal berechnet.

| Fenster | y [µm] | Zellen | Körner | Zonengrenzen [µm] |
|---|---|---|---|---|
| Band 1 oben | 0 … 886 | 216 975 | 12 928 | 837 / 1036 |
| Band 2 | 886 … 1772 | 216 975 | 9 174 | 736 / 1326 |
| Band 3 | 1772 … 2658 | 216 975 | 5 266 | 551 / 1677 |
| Band 4 unten | 2658 … 3544 | 216 975 | 3 132 | 898 / 1424 |
| Referenzband | 1339 … 2225 | 216 975 | 7 601 | 667 / 1495 (Markierung) |
| volle Höhe | 0 … 3698 | 905 025 | 30 673 | 814 / 1552 |

: Auswertefenster der Höhenstudie. Alle mit x = 30 … 2811 µm und einer Zelle je EBSD-Pixel (3,371 µm). Das Fenster über die volle Höhe enthält viermal so viele Körner wie das Referenzband.\label{tbl:studycases}

![(a) Die Auswertefenster im EBSD-Scan: vier gestapelte Bänder (schwarz) und das Referenzband aus Kapitel 2 (weiß gestrichelt); grün die je Band bestimmten Zonengrenzen. Rechts außerhalb aller Fenster liegt homogenes 316L, das bewusst nicht mitmodelliert wird. (b) Lage und Breite der Übergangszone über die Probenhöhe. Die Grenzfläche ist nicht ortsfest: sie wandert um mehrere hundert µm, und die Zonenbreite schwankt um den Faktor fünf.](fig_study_map.png){#fig:studymap width=16cm}

## 3.2  Zonengrenzen aus der Mikrostruktur

Eine eingezeichnete Markierung gibt es nur für das Referenzband. Für die übrigen Fenster werden die Zonengrenzen daher aus dem Gefüge selbst bestimmt, mit zwei unabhängigen Signalen:

- **Grenze 17-4PH | Übergang:** die Halbwertsstelle des spaltenweisen BCC-Anteils (links Plateau bei 0,6 … 0,85, rechts null).
- **Grenze Übergang | 316L:** die Stelle, an der der Flächenanteil grober Körner (d_eq > 500 µm) 80 % übersteigt — der Beginn des kolumnaren 316L-Gefüges.

Beide Schwellen sind am Referenzband kalibriert: dort liefert die Vorschrift 709 / 1394 µm gegenüber den markierten 667 / 1495 µm, also +43 und −101 µm Abweichung (1,5 bzw. 3,6 % der Fensterbreite). Das ist die Genauigkeit, mit der die Grenzen der übrigen Fenster zu verstehen sind.

Das Ergebnis ist selbst schon ein Befund (Abb.\ \ref{fig:studymap}b): die Grenzfläche ist nicht ortsfest. Die linke Grenze wandert zwischen 551 und 898 µm, die rechte zwischen 1036 und 1677 µm; die Breite der Übergangszone schwankt zwischen 199 µm (Band 1) und 1126 µm (Band 3). Über die volle Höhe gemittelt ergeben sich 814 / 1552 µm. Eine Steifigkeitsmessung, die über die volle Probenbreite mittelt — wie die DIC —, mittelt also zwangsläufig auch über diese Wanderung hinweg.

Weil die Zonenzuordnung bei s ≡ 1 nur festlegt, welche Zellen zu welchem Mittelwert zusammengefasst werden (die Konstanten aller drei Werkstoffe sind identisch, siehe Abschnitt 7.6), lässt sich ihr Einfluss sauber abtrennen: Rechnet man alle Fenster stattdessen mit den festen Grenzen des Referenzbandes, ändert sich die Voigt-Schranke der Übergangszone um höchstens 14 GPa (Band 1: 166,5 → 152,3 GPa), meist um weniger als 5 GPa. Keine der folgenden Aussagen hängt an der Zonendefinition.

## 3.3  Was ohne FE-Rechnung bereits feststeht

Für jedes Fenster lassen sich Voigt- und Reuss-Schranke der Übergangszone direkt aus den kornweisen Tensoren berechnen — solverunabhängig, wie in Abschnitt 2.2. Alle sechs Fenster wurden zusätzlich mit dolfinx gerechnet (Abschnitt 3.6); die FE-Werte stehen mit in der Tabelle.

| Fenster | A_Ü | Reuss | Schätzer | FE | Voigt | Mw./Voigt | s_erf |
|---|---|---|---|---|---|---|---|
| Band 1 oben | 0,07 | 131,5 | 150,6 | 142,9 | 166,5 | 1,40 | 1,63 |
| Band 2 | 0,21 | 159,0 | 170,5 | 169,4 | 198,8 | 1,17 | 1,37 |
| Band 3 | 0,41 | 170,0 | 185,7 | 181,5 | 210,8 | 1,10 | 1,28 |
| Band 4 unten | 0,19 | 134,5 | 144,0 | 140,1 | 154,9 | 1,50 | 1,66 |
| Referenz­band | 0,30 | 164,2 | 174,4 | 175,2 | 203,5 | 1,14 | 1,33 |
| volle Höhe | 0,27 | 143,6 | 157,5 | 156,6 | 182,4 | 1,27 | 1,48 |

: Übergangszone in allen Fenstern, E_x [GPa]. A_Ü = Flächenanteil der Übergangszone, FE = dolfinx, Mw. = Messwert. Der gemessene Grenzflächenwert beträgt 232,4 ± 13 GPa und liegt in jedem Fenster oberhalb der Voigt-Schranke — und damit erst recht oberhalb der FE-Lösung. s_erf = 232,4 / E_FE.\label{tbl:studytrans}

Tabelle\ \ref{tbl:studytrans} fasst die Werte zusammen, Abb.\ \ref{fig:studybounds}a zeigt sie. Das Ergebnis ist eindeutig: In keinem einzigen Fenster erreicht die Voigt-Schranke den Messwert. Der Abstand beträgt 10 % (Band 3) bis 50 % (Band 4); das Referenzband aus Kapitel 2 liegt mit 14 % nicht etwa günstig, sondern es hat von allen Bändern die zweithöchste Voigt-Schranke — die dort getroffene Aussage ist also eher konservativ. Über die volle Probenhöhe, dem der DIC-Messung am nächsten kommenden Fenster, wächst der Abstand auf 27 %, und der erforderliche Vorfaktor steigt von s ≈ 1,33 auf s ≈ 1,48. Die FE-Lösung liegt in jedem Fenster innerhalb ihrer Schranken und in fünf von sechs Fenstern unter dem Schätzer — der Abstand zum Messwert wird durch die Feldrechnung also eher größer, nicht kleiner.

Der Grund für die Streuung zwischen den Bändern ist derselbe wie in Abschnitt 2.2: Der Übergangsbereich ist FCC-dominiert, und FCC-Austenit ist in seiner weichen Richtung mit E⟨100⟩ = 94 GPa sehr nachgiebig. Wo der BCC-Restanteil höher ist (Band 1: 16,5 %, Band 2: 12,8 %), liegt die Schranke höher; wo er praktisch verschwindet (Band 4: 3,6 %), fällt sie ab. Bemerkenswert ist, dass gerade das Band mit der breitesten Übergangszone (Band 3, 1126 µm) auch die höchste Schranke hat — dort ist der Anteil grober, günstig orientierter Körner am größten.

![Voigt-/Reuss-Bereich (violett), arithmetischer Schätzer (offener Kreis) und FE-Lösung (Raute) je Auswertefenster gegen den Messwert (blau, mit Streuband). (a) Übergangszone: der Messwert 232,4 GPa liegt in jedem Fenster oberhalb der Voigt-Schranke. (b) 316L-Zone: im Referenzband und in den Bändern 2 und 3 entartet der Schrankenbereich, weil dort ein einziges Korn dominiert; erst über die volle Höhe entsteht ein echter Aggregatwert, und der trifft den Messwert.](fig_study_bounds.png){#fig:studybounds width=16cm}

## 3.4  Der 316L-Bereich über die volle Höhe: ein echter Aggregatwert

Die zweite offene Frage aus Abschnitt 3.1 beantwortet sich mit dem Fenster über die volle Höhe. Dort enthält die 316L-Zone 388 Körner statt 18, und das größte nimmt nur noch 57 % statt 99,8 % der Zonenfläche ein — kein Einkristall mehr, aber immer noch grobkörnig.

| Fenster | 17-4PH: FE | 316L: Körner | größtes Korn | 316L: Reuss / Schätzer / FE / Voigt |
|---|---|---|---|---|
| Referenzband | 187,0 | 18 | 99,8 % | 185,8 / 185,8 / 186,0 / 186,0 |
| volle Höhe | 172,4 | 388 | 57,2 % | 152,6 / 161,4 / 160,2 / 182,0 |
| *Messwert* | *201,7 ± 2* | | | *162,7 ± 2* |

: Monobereiche im Referenzband und über die volle Probenhöhe, E_x [GPa]. FE = dolfinx.\label{tbl:studymono}

Der Modul der 316L-Zone fällt damit (Tabelle\ \ref{tbl:studymono}) von 186,0 auf 160,2 GPa und liegt nur noch 1,6 % neben dem gemessenen Wert von 162,7 GPa (Abb.\ \ref{fig:studybounds}b). Die in Kapitel 4 als „erwartbar" bezeichnete Abweichung des Referenzbandes war also tatsächlich ein reiner Stichprobeneffekt und kein Modellfehler — sobald genügend Körner erfasst werden, trifft das Modell den Monowerkstoff.

Das ist der erste belastbare Validierungspunkt dieser Arbeit und stärkt den Kernbefund erheblich: dasselbe Modell, dieselben Konstanten und dieselbe Idealisierung, die den 316L-Bereich auf 1,6 % genau treffen, verfehlen den Übergangsbereich um 33 % (156,6 gegenüber 232,4 GPa). Der Fehler liegt damit sehr wahrscheinlich nicht im Modell, sondern in der Annahme, der Übergangswerkstoff habe dieselben Einkristallkonstanten wie seine Ausgangswerkstoffe.

Für 17-4PH bleibt eine Lücke: 172,4 gegenüber gemessenen 201,7 GPa, also 14 % zu weich — im Referenzband waren es 7 %. Das Vorzeichen ist hier bemerkenswert, denn es ist dasselbe wie beim Übergangsbereich, nur kleiner. Zwei Erklärungen liegen nahe und sind mit den vorliegenden Daten nicht zu trennen: der BCC-Anteil der 17-4PH-Zone sinkt über die volle Höhe auf 48 % (Referenzband: 67 %), sodass mehr weicher Austenit einfließt; und die α-Eisen-Konstanten des BCC-Satzes enthalten die Ausscheidungshärtung des 17-4PH nicht (Anhang E).

## 3.5  Streifenprofile aller Fenster

Abb.\ \ref{fig:studyprofiles} zeigt die E(x)-Profile aller sechs Fenster nach der Vorschrift aus Anhang B, jeweils mit dem zugehörigen Voigt-/Reuss-Band. Drei Beobachtungen:

- **Das Minimum liegt überall in oder am Rand der Übergangszone**, nie in einem Monobereich. In Band 1 fällt die FE-Lösung bis auf 111 GPa ab — der schärfste Einbruch der ganzen Karte.
- **Kein Streifen in keinem Fenster** erreicht mit seiner Voigt-Schranke das Messband um 232,4 GPa. Das höchste Streifenmaximum überhaupt liegt bei rund 215 GPa (Voigt-Schranke) bzw. 202 GPa (FE), beides in Band 3 am rechten Rand der Übergangszone.
- **Der 316L-Teil ist nur im Referenzband und in Band 2/3 flach** (ein Korn); in Band 1 und 4 zeigt er dieselbe Struktur wie ein echter Polykristall, und über die volle Höhe verläuft das Profil deutlich glatter, weil über 1097 statt 263 Zellzeilen gemittelt wird.

![Streifenprofile E(x) aller sechs Auswertefenster (Streifenbreite ≈ 100 µm, Vorschrift wie in Anhang B). Schwarz die FE-Lösung (dolfinx), rot der arithmetische Schätzer, violett der Voigt-/Reuss-Bereich desselben Streifens, gestrichelt die Zonengrenzen des jeweiligen Fensters, blau der gemessene Grenzflächenwert. In keinem Streifen berührt die obere Schranke das Messband.](fig_study_profiles.png){#fig:studyprofiles width=16cm}

## 3.6  FE-Bestätigung und was sie über den Schätzer sagt

Alle sechs Fenster wurden mit dolfinx gerechnet (`run_study.sh`, s(x) ≡ 1, 8 MPI-Ranks). Der Fall über die volle Höhe ist mit 905 025 Zellen und 1,81 Mio. Freiheitsgraden rund viermal so groß wie der Referenzfall; der Löser (CG mit GAMG-Vorkonditionierung) brauchte in allen Fällen 42 bis 45 Iterationen — die Rechenzeit skaliert praktisch linear mit der Zellzahl.

Zwei Prüfungen bestätigen die Konsistenz:

- **Schrankentest.** Alle 18 Zonenmoduln (6 Fenster × 3 Zonen) liegen innerhalb ihrer eigenen Voigt-/Reuss-Schranken. Der Kernbefund ist damit doppelt abgesichert: solverunabhängig durch die Schranke und direkt durch die Feldlösung.
- **Gleichgewichtstest.** ⟨σ_xx⟩ ist in jedem Fenster über alle drei Zonen identisch (Beispiel volle Höhe: 0,1624262272 GPa in allen drei Zonen).

Und sie beantworten eine Frage, die Anhang C bisher nur am Referenzband klären konnte: wie gut ist der arithmetische Schätzer? Über alle 18 Zonenwerte liegt er im Mittel 2,0 GPa über der FE-Lösung, maximal 7,7 GPa (5 %, Band 1); in 15 von 18 Fällen ist die FE-Lösung die weichere. Auf Streifenebene beträgt die größte Einzelabweichung 9,7 GPa. Die in Abschnitt 3.3 vor der Rechnung getroffene Vorhersage — FE etwas unter dem Schätzer, Abstand zum Messwert dadurch eher größer — hat sich also bestätigt. Für eine Vorabschätzung ohne Löser ist der Schätzer damit auch über sehr verschieden aufgebaute Fenster hinweg brauchbar; für lokale Größen bleibt er es nicht (Abschnitt 2.5).

# 4  Diskussion

**Die Mikrostruktur erklärt die Überhöhung nicht.** Das Modell reproduziert nicht nur den Betrag nicht, es kehrt das Vorzeichen um: der Übergangsbereich ist rechnerisch die weichste Zone (175 GPa gegenüber 187 und 186). Entscheidend ist, dass dieser Befund nicht von der FE-Rechnung abhängt — bereits die Voigt-Schranke der Zone liegt mit 203,5 GPa 14 % unter dem Messwert — und dass er nicht am gewählten Ausschnitt hängt: in allen sechs Auswertefenstern der Höhenstudie liegt der Messwert oberhalb der Voigt-Schranke, über die volle Probenhöhe sogar um 27 % (Kapitel 3). Auch die Idealisierungsrichtung stützt die Aussage: der gewählte ebene Spannungszustand ist die weichere der beiden 2D-Näherungen, die steifere Alternative (ebene Dehnung) würde die Werte anheben, aber sie wäre für die 2 mm dicke, in Dickenrichtung freie Probe physikalisch falsch und läge zudem ebenfalls unterhalb des Messwerts, da die Voigt-Schranke die Obergrenze markiert.

**Die Antwort auf die Ausgangsfrage lautet damit: die Konstanten müssen angepasst werden** — um etwa den Faktor 1,33. Materialkundlich ist das begründungsbedürftig; naheliegende Kandidaten sind Ausscheidungshärtung (17-4PH ist ein ausscheidungshärtender Stahl, Cu- und Nb-Anteile), veränderte Phasenanteile gegenüber dem hier ausgewerteten Schliff sowie ein durch Aufmischung entstandener Mischkristall mit anderen Einkristallkonstanten. Ein Faktor 1,33 auf den gesamten Steifigkeitstensor ist allerdings groß: die Einkristallkonstanten von Stählen variieren zwischen Austenit und Martensit nur um etwa 10 %.

**Alternative Erklärungen, die vor dieser Schlussfolgerung zu prüfen sind.** Die Diskrepanz kann auch ganz oder teilweise aus dem Vergleich selbst stammen:

- *Unterschiedliche Auswertevolumina.* Das experimentelle Fenster „Grenzfläche" ist mehrere Millimeter breit, die modellierte Übergangszone nur 0,83 mm. Der Messwert mittelt also über deutlich mehr Material, insbesondere über Anteile beider Monobereiche. Dass der 14-Punkt-DIC-Verlauf an einem einzelnen Punkt auf 237 GPa steigt, spricht allerdings dafür, dass die Überhöhung real lokal ist.
- *Repräsentativität des Schliffs.* Der EBSD-Scan stammt nicht aus derselben Probe wie der Zugversuch, und die Zuordnung ROI ↔ DIC-Messpunkt ist nicht bekannt. Insbesondere ist unklar, ob der gescannte Bereich denselben Aufmischungsgrad hat wie die geprüfte Grenzfläche. Die *innere* Repräsentativität — die Frage, ob der ausgewertete Ausschnitt für den Scan steht — ist mit Kapitel 3 dagegen geklärt: das Ergebnis ändert sich über die Probenhöhe nur im Betrag, nicht in der Aussage.
- *Zweidimensionalität.* Modelliert wird eine Schnittebene; die Steifigkeitsbeiträge der darunter- und darüberliegenden Körner fehlen. Bei den sehr großen 316L-Körnern (Äquivalenzdurchmesser über 1 mm gegenüber 2 mm Probendicke) ist die Annahme, die Schnittebene sei repräsentativ für die Dicke, kaum belastbar.
- *Einkristallkonstanten.* Alle Absolutwerte skalieren mit Tabelle\ \ref{tbl:singlecrystal}. Die dort genutzten Werte gelten für einen 304-Typ-Austenit bzw. reines α-Eisen und sind nicht für 316L, 17-4PH oder den Übergangswerkstoff validiert (Anhang E). Der α-Eisen-Satz ist gegen publizierte Messungen abgesichert, der Austenit-Satz nicht gegen die Primärquelle geprüft.

**Die Monobereiche: im Referenzband nicht verwertbar, über die volle Höhe dagegen schon.** Der 316L-Bereich des Referenz-ROI besteht zu 99,8 % aus einem einzigen Korn — sein Modulwert von 186 GPa ist der Richtungsmodul einer einzelnen Orientierung, während der Messwert von 162,7 GPa ein Aggregatmittel über viele Körner ist. Die Abweichung von +14 % war damit erwartbar und kein Modellfehler; Abschnitt 3.4 bestätigt das quantitativ: über die volle Probenhöhe (388 Körner) fällt der Wert auf 160,2 GPa und trifft den Messwert auf 1,6 %. Für 17-4PH ist der Vergleich schon im Referenzband aussagekräftiger (5771 Körner, kein dominierendes Einzelkorn): 187,0 gegenüber 201,7 GPa, also 7 % zu weich; über die volle Höhe wächst die Lücke auf 14 %, weil dort der BCC-Anteil der Zone von 67 auf 48 % sinkt. Eine plausible Größenordnung bleibt es für Platzhalterkonstanten ohne Ausscheidungshärtung und für die ebene Idealisierung — aber sie zeigt, dass das Modell die martensitische Seite systematisch etwas zu weich abbildet.

**Dasselbe gilt für den scheinbaren Querkontraktionskoeffizienten** ν_xy = 0,058 aus Tabelle\ \ref{tbl:fe}, der weit unter jedem Stahl-Literaturwert liegt. Er ist kein Fehler und kein Werkstoffkennwert, sondern ebenfalls eine Folge des dominanten 316L-Korns: dessen Orientierung (φ1, Φ, φ2 = 141°, 88°, 359°) legt eine ⟨100⟩-Richtung fast in die Schliffebene, und für diese Lage ist der kubische Einkristall in-plane auxetisch — die Einzelkornwerte sind ν_xy = −0,18 und ν_xz = +0,79. Die Querkontraktion findet also fast vollständig in Dickenrichtung statt statt in der Ebene. Die Summe ν_xy + ν_xz = 0,61 ist dagegen völlig normal (isotrop wären es 2 × 0,30). In den feinkörnigen Zonen ergibt die Mittelung über viele Orientierungen erwartungsgemäß fast isotrope Werte (17-4PH: ν_xy = 0,30, ν_xz = 0,32). Der ROI-Mittelwert ist damit vom einen großen Korn dominiert und nicht mit dem experimentellen ν = 0,427 vergleichbar.

**Nächste Schritte zum Ferritbefund.** Aus Abschnitt 2.6 ergeben sich zwei konkrete Ansatzpunkte:

1. **s(x) an den Ferritverlauf koppeln**, statt es frei anzupassen — etwa s(x) = 1 + a·|dF/dx| oder als Funktion des Ferritgehalts. Ein einziger Parameter a ließe sich am gemessenen E-Verlauf kalibrieren und wäre damit prüfbar statt beliebig.
2. **Gefügeanalyse in der Zone des steilsten Gradienten** (Messpunkte 4–6): Welche Phasen liegen dort vor, gibt es Ausscheidungen, und sind deren Einkristallkonstanten hoch genug, um den Faktor 1,33 zu tragen? Das ist die materialkundliche Frage, die diese Arbeit offen lassen muss.

**Verhältnis zur Neper-Studie (069).** Beide Arbeiten teilen Materialmodell und Konventionen (Bond-Rotation, Voigt-Reihenfolge, Konstantensatz), unterscheiden sich aber grundsätzlich in der Fragestellung: 069 bestimmt effektive Kennwerte statistisch äquivalenter Gefüge, 070 prüft eine ortsfeste Hypothese an einem konkreten Bereich. Die dort diskutierte Frage nach dem Anteil der Kornform an der Anisotropie stellt sich hier nicht — die Kornform ist nicht modelliert, sondern gemessen und direkt übernommen.

# 5  Schlussfolgerungen

1. Die im Übergangsbereich gemessene Steifigkeitsüberhöhung (232,4 ± 13 GPa) ist mit Kornstruktur und Kristallorientierung allein nicht erklärbar. Das Modell liefert dort 175,2 GPa und macht den Übergang zur weichsten statt zur steifsten Zone.
2. Diese Aussage ist stärker als ein Simulationsergebnis: die Voigt-Schranke derselben Kornstruktur beträgt 203,5 GPa und liegt damit 14 % unter dem Messwert. Keine Anordnung dieser Körner erreicht mit diesen Konstanten den gemessenen Wert.
3. Sie hängt auch nicht am gewählten Ausschnitt. In allen sechs Auswertefenstern der Höhenstudie — vier gestapelten Bändern, dem Referenzband und der vollen Probenhöhe, alle mit dolfinx gerechnet — liegt der Messwert oberhalb der Voigt-Schranke, mit 10 bis 50 % Abstand. Das Referenzband aus Kapitel 2 ist dabei eines der günstigsten Fenster; über die volle Höhe wächst der Abstand auf 27 %.
4. Der Steifigkeitstensor des Übergangswerkstoffs müsste um den Faktor s ≈ 1,33 (Referenzband) bis s ≈ 1,48 (volle Probenhöhe) über dem der Ausgangswerkstoffe liegen; die einzelnen Bänder verlangen zwischen 1,28 und 1,66. Das ist ein großer Betrag und materialkundlich zu begründen.
5. Über die volle Probenhöhe entsteht der erste echte Validierungspunkt: die 316L-Zone enthält dort 388 Körner statt eines einzigen, und ihr FE-Modul trifft mit 160,2 GPa den gemessenen Wert von 162,7 GPa auf 1,6 %. Dasselbe Modell verfehlt den Übergangsbereich um 33 % — die Diskrepanz liegt also im Werkstoff, nicht im Modell.
6. Ein Nebenbefund der Höhenstudie: die Grenzfläche ist nicht ortsfest. Ihre Lage wandert über die Probenhöhe um mehrere hundert µm, ihre Breite schwankt zwischen 199 und 1126 µm. Eine über die Probenbreite mittelnde Messung mittelt zwangsläufig auch darüber.
7. Vor dieser Schlussfolgerung sind die Vergleichbarkeitsfragen aus Kapitel 4 zu klären — insbesondere die unterschiedlichen Auswertevolumina und die Repräsentativität des Schliffs für die geprüfte Probe.
8. Für den makroskopischen Kennwert reicht eine arithmetische Mittelung der kornweisen Moduln aus (Abweichung zur FE-Lösung im Mittel unter 0,1 GPa). Der Mehrwert der expliziten Feldrechnung liegt im lokalen Feld: Lastumlagerung, Querdehnungsbehinderung und Spannungsüberhöhungen bis zum 2,1-fachen des Mittelwerts (Abschnitt 2.5) — die Ausgangsgrößen für alle Folgefragen zu Ermüdung und Rissinitiierung.
9. Eine unabhängige, an derselben Probe gemessene Größe zeigt in dieselbe Richtung: das Steifigkeitsmaximum fällt mit dem steilsten Gradienten des Ferritgehalts zusammen (Abschnitt 2.6). Damit ließe sich s(x) an eine Messgröße koppeln, statt es frei anzupassen.
10. Nächste sinnvolle Schritte: validierte Einkristallkonstanten statt der Platzhalter; ein per-Messpunkt-EBSD-Export für eine exakte Kornkarte; ein Scan aus der geprüften Probe; sowie eine Anpassung des Profils s(x) an den gemessenen DIC-Verlauf, um die räumliche Form der Versteifung zu charakterisieren.

# 6  Annahmen und Grenzen

- **Einkristallkonstanten sind Literaturwerte verwandter Werkstoffe**, nicht für 316L bzw. 17-4PH gemessen (`config.json`, identisch zu 069; Herkunft und Belastbarkeit in Anhang E). Alle Absolutwerte skalieren mit ihnen. Der Übergangswerkstoff hat einen eigenen, editierbaren Konstantensatz, voreingestellt auf die Werte der Ausgangswerkstoffe.
- **Kornkarte rekonstruiert, nicht gemessen** (kein Rohdatenexport pro Messpunkt verfügbar). Güte: mittlerer Farbfehler 16,7/255 gegen das Originalbild. Innerhalb eines Korns wird die mittlere Orientierung des Korns angesetzt; Orientierungsgradienten (Subkörner, Lath-Struktur) sind nicht abgebildet.
- **Ebener Spannungszustand:** ein Schliff, keine Information über die Dickenrichtung. Bei Korngrößen über 1 mm gegenüber 2 mm Probendicke ist die Repräsentativität der Schnittebene fraglich. Die Out-of-plane-Dehnung ist zellweise unstetig und damit streng genommen inkompatibel (Abschnitt 7.5).
- **Zonengrenzen sind scharf** und wurden aus der eingezeichneten Markierung übernommen; real ist der Übergang graduell. Der Vorfaktor s(x) kann dies abfangen.
- **Reguläres Netz:** Korngrenzen sind auf ein Pixel (3,371 µm) genau treppenförmig approximiert. Für die effektive Steifigkeit unkritisch (Netzstudie, Anhang D), für lokale Spannungsspitzen an Korngrenzen relevant.
- **Lineare Elastizität, kleine Verzerrungen,** homogene Einkristalle je Korn, ideale Kornbindung, keine Korngrenzen- oder Eigenspannungseffekte. Die gemessenen Kennwerte stammen aus einem Zugversuch bis 1,5 % Dehnung; die Steifigkeitsauswertung ist dort noch elastisch.
- **Der 316L-Bereich des Referenzbandes ist kein Polykristall** (99,8 % ein Korn) und daher nicht für einen Aggregatvergleich geeignet. Über die volle Probenhöhe (388 Körner, größtes Korn 57 %) entfällt diese Einschränkung weitgehend, das Gefüge bleibt aber grobkörnig.
- **Die Zonengrenzen der Höhenstudie** sind aus dem Gefüge bestimmt, nicht markiert; die Kalibrierung am Referenzband zeigt eine Unsicherheit von rund 100 µm. Auf die Ergebnisse wirkt sie sich kaum aus (Abschnitt 3.2).
- **Die Bänder 1 und 4** liegen am oberen und unteren Rand des Scans; ihr „316L-Bereich" ist dort deutlich weicher (121 bzw. 140 GPa) als in der Kartenmitte. Ob das Gefüge oder die Scanberandung dahintersteckt, ist mit diesem Datensatz nicht zu entscheiden.
- **Vergleichsbasis:** experimentelle Auswertefenster (mehrere mm) und modellierte Zonen (0,83 mm) sind nicht deckungsgleich; EBSD-Scan und Zugprobe sind nicht identisch.

# 7  Modell und Methodik im Detail

## 7.1  Datengrundlage

| Datei | Inhalt |
|---|---|
| `WAAM_N=1_A12D_Uebergangsbereich.txt` | TSL/OIM-Kornexport, 44 Spalten, 45 451 gültige Körner: Phase, mittlere Bunge-Winkel, Schwerpunkt, Ellipsenfit, Fläche |
| `WAAM_N=1_A12D_Uebergangsbereich.bmp` | IPF-Z-Darstellung derselben Karte, 1393 × 1097 px bei 3,371 µm/px (4696 × 3698 µm) |
| `..._mit_AR_Bereich.bmp` | dieselbe Karte mit eingezeichneter Aufbaurichtung, Auswerte-ROI (schwarz) und Übergangszone (grün) |
| `Kennwerte_Zugversuche_WAAM_N=1 (2).pdf` | Zugversuchskennwerte je Auswertefenster (extern gemessen) |
| `Verlauf_Elastizitätsmodul_lokal_N=1.xlsx` | lokaler E-Verlauf aus DIC, 14 Messpunkte, Probe MHo1030_A9D_5 |
| `Ferritscope_WAAM_N=1.xlsx` | Ferritgehalt an denselben 14 Messpunkten derselben Probe (Abschnitt 2.6) |

Die Probe ist in Orientierung V geprüft (Lastachse senkrecht zu den Layerebenen, also ∥ Aufbaurichtung), Probendicke 2 mm; der Übergang liegt in der Mitte der Messstrecke. Im EBSD-Bild zeigt der Aufbaurichtungspfeil nach −x, die Lastachse ist damit die horizontale Bildachse x (Abb.\ \ref{fig:roimap}).

## 7.2  Rekonstruktion der Kornkarte

### 7.2.1  Problemstellung

Für eine explizite Modellierung wird pro Bildpunkt die Kornzugehörigkeit und die Orientierung benötigt. Ein Rohdatenexport pro Messpunkt (`.ang`/`.ctf`) lag für diesen Scan nicht vor — verfügbar sind nur der Kornexport (eine Zeile je Korn: Schwerpunkt, mittlere Orientierung, Ellipsenfit, Fläche, Phase) und das gerenderte IPF-Bild. Die Kornkarte wird daher aus beiden gemeinsam rekonstruiert.

### 7.2.2  IPF-Farbmodell und Referenzrichtung

Die im Kornexport enthaltenen RGB-Werte sind eine willkürliche Zufallspalette und stimmen nicht mit dem Bild überein. Stattdessen wird die IPF-Farbe jedes Korns aus dessen Eulerwinkeln selbst berechnet: Die Referenzrichtung wird in den Kristallrahmen transformiert, d = g·e_ref, mit den 24 kubischen Symmetrieoperationen in das Standard-Grunddreieck (z ≥ x ≥ y ≥ 0) reduziert und nach der üblichen TSL-Vorschrift eingefärbt,

$$ (R,G,B) \propto (z-x,\; x-y,\; y), \qquad \text{normiert auf } \max = 1, \text{ Gamma } \gamma = \tfrac12 . $$

Die Referenzrichtung wurde durch Abgleich mit dem Bild bestimmt: für e_ref = z (Schliffnormale) beträgt der mediane Farbfehler an den Kornschwerpunkten 20,7 (von 255), für x und y dagegen 78 bzw. 64. Die Karte ist also eine IPF-Z-Karte; das ist eine Messgröße, keine Annahme.

### 7.2.3  Zuordnung Pixel → Korn

Jedes Pixel wird demjenigen Korn zugeordnet, das die Kostenfunktion

$$ \text{cost} = \max\!\left(\sqrt{(u/a)^2+(v/b)^2}-1,\;0\right) \;+\; \lambda\,\frac{\lVert \mathrm{RGB}_{\text{Bild}}-\mathrm{RGB}_{\text{IPF}}(\text{Korn})\rVert_1/3}{85} $$

minimiert. Der erste Term ist der Abstand zum Ellipsenfit des Korns (u, v = Pixelkoordinaten im gedrehten Ellipsensystem, a, b = Halbachsen), als *Hinge* formuliert: innerhalb der Ellipse kostenfrei, außerhalb linear wachsend. Der zweite Term bestraft die Farbabweichung; λ = 3. Kandidaten sind die 30 nächstgelegenen Kornschwerpunkte plus die 120 flächengrößten Körner der Karte — letztere sind nötig, weil die stark gekrümmten Chevron-Körner des 316L ihren eigenen Schwerpunkt teilweise gar nicht enthalten.

Güte der Rekonstruktion: mittlerer Farbfehler 16,7 von 255 gegenüber dem Originalbild (ohne Beschriftungspixel), entsprechend 6,5 % — die rekonstruierte Karte gibt Kornform, Korngrenzen und Orientierungsverteilung des Scans wieder (Abb.\ \ref{fig:recon}).

![Vergleich Originalscan (a) und rekonstruierte Kornkarte (b) im modellierten ROI, beide IPF-Z. Gestrichelt: Grenzen der Übergangszone; schwarzes Rechteck in (b): Lage des Netzausschnitts aus Abb.\ \ref{fig:meshzoom}. Mittlerer Farbfehler 16,7 von 255.](fig_reconstruction.png){#fig:recon width=15cm}

### 7.2.4  ROI und Zonen

Die Rechteckkoordinaten (Tabelle\ \ref{tbl:roi}) wurden pixelgenau aus der annotierten Bitmap ausgelesen (Mittellinie der gezeichneten Striche, Maßstabsfaktor 3142/1393 = 2,2555 gegenüber dem Originalbild):

| Größe | Wert (Kartenrahmen) |
|---|---|
| ROI | x = 30 … 2811 µm, y = 1339 … 2225 µm → 2781 × 886 µm |
| Zone 0 (17-4PH) | x < 666,6 µm (Breite 637 µm, Flächenanteil 0,229) |
| Zone 1 (Übergang) | 666,6 ≤ x < 1494,5 µm (Breite 828 µm, Flächenanteil 0,297) |
| Zone 2 (316L) | x ≥ 1494,5 µm (Breite 1318 µm, Flächenanteil 0,474) |

: Auswertebereich und Zonengrenzen, ausgelesen aus `..._mit_AR_Bereich.bmp`.\label{tbl:roi}

## 7.3  Reguläres Netz statt geometrischer Vernetzung

Die Kornstruktur wird nicht geometrisch vernetzt (keine an Korngrenzen ausgerichteten Elemente), sondern als Feld auf einem regulären Netz abgebildet: ein Q1-Viereckelement je EBSD-Pixel, Kantenlänge 3,371 µm, 263 × 825 = 216 975 Elemente, 218 064 Knoten, 436 128 Freiheitsgrade. Jede Zelle erhält die Orientierung, die Phase und den Zonenindex des Korns, in dem ihr Mittelpunkt liegt.

*Sprachregelung:* „Zelle" bezeichnet ein Feld des Mikrostrukturgitters, „Element" das zugehörige finite Element — es ist dasselbe Objekt, die beiden Begriffe werden synonym verwendet.

Der Vorteil ist Robustheit und exakte Übertragbarkeit des Scans (kein Vernetzungsfehler bei den extrem feinen Martensitkörnern, deren Äquivalenzdurchmesser teilweise nur wenige Pixel beträgt). Der Preis ist eine treppenförmige Approximation der Korngrenzen mit einer Auflösung von einem Pixel; für die *effektive* Steifigkeit ist das unkritisch (Netzstudie, Anhang D), für lokale Spannungsspitzen direkt an Korngrenzen dagegen relevant.

Abb.\ \ref{fig:meshzoom} zeigt das Prinzip an einem Ausschnitt von 42 × 30 Elementen (142 × 101 µm) aus der Übergangszone; seine Lage im ROI ist in Abb.\ \ref{fig:recon}b eingezeichnet. Über dem gesamten ROI ließen sich die Elementkanten nicht mehr auflösen, daher die Vergrößerung. Gut erkennbar: die Elementkanten (dünn) folgen dem starren Raster, während die Korngrenzen (dick) diesem Raster treppenförmig folgen, statt geometrisch nachgebildet zu werden. Jedes Element trägt genau eine Orientierung, nämlich die mittlere Orientierung des Korns, in dem sein Mittelpunkt liegt.

![Das reguläre Netz über der Kornstruktur, Ausschnitt von 42 × 30 Elementen aus der Übergangszone (Lage in Abb.\ \ref{fig:recon}b markiert). (a) EBSD-Originalscan des Ausschnitts. (b) Modelleingang: ein Q1-Element je EBSD-Pixel; dünne Linien sind Elementkanten, dicke Linien die aus den Korn-IDs rekonstruierten Korngrenzen. Diese verlaufen treppenförmig entlang der Elementkanten — das Netz wird nicht an die Kornform angepasst.](fig_mesh_zoom.png){#fig:meshzoom width=15cm}

## 7.4  Kornweise Steifigkeit

Jedem Korn wird der kubische Einkristalltensor seiner Phase zugewiesen und über die 6 × 6-Bond-Rotation in den Probenrahmen gedreht:

$$ \mathbf{C}_{\text{Probe}} = \mathbf{M}(g)\,\mathbf{C}_{\text{Kristall}}\,\mathbf{M}(g)^{\mathsf T},
\qquad \mathbf{M} = \mathbf{M}(a),\; a = g^{\mathsf T}, $$

mit g der Rotation aus den Bunge-Winkeln (v_Kristall = g·v_Probe) und Voigt-Reihenfolge (xx, yy, zz, yz, xz, xy) mit technischen Gleitungen. Der kubische Tensor ist durch drei Konstanten bestimmt:

$$ C_{11}=C_{22}=C_{33},\quad C_{12}=C_{13}=C_{23},\quad C_{44}=C_{55}=C_{66},\quad \text{sonst } 0 . $$

Da g je Korn verschieden ist, hat jedes Korn einen eigenen Steifigkeitstensor, auch bei gleicher Phase und gleichem Werkstoff. Warum die Rotationsgleichung gerade diese Form hat — insbesondere warum dort eine Transponierte und keine Inverse steht, obwohl die Transformationsmatrix nicht orthogonal ist — ist in Anhang A hergeleitet.

Die verwendeten Einkristallkonstanten und die daraus folgenden Richtungsmoduln stehen in Tabelle\ \ref{tbl:singlecrystal}; ihre Herkunft in Anhang E.

## 7.5  Ebener Spannungszustand: statische Kondensation

Es liegt nur ein Schliff vor; gerechnet wird daher im ebenen Spannungszustand. Der rotierte 3D-Tensor ist im Allgemeinen triklin (keine Nullen an den Kopplungsstellen), eine der üblichen Kurzformeln für plane stress ist damit nicht anwendbar. Stattdessen wird exakt statisch kondensiert: Mit der Aufteilung der Voigt-Indizes in die In-plane-Gruppe A = (xx, yy, xy) und die Out-of-plane-Gruppe B = (zz, yz, xz) und der Forderung

$$ \underline\sigma_B = \mathbf{0} \quad\Longrightarrow\quad \underline\varepsilon_B = -\mathbf{C}_{BB}^{-1}\mathbf{C}_{BA}\,\underline\varepsilon_A $$

folgt

$$ \boxed{\;\mathbf{C}^{\text{PS}} = \mathbf{C}_{AA} - \mathbf{C}_{AB}\,\mathbf{C}_{BB}^{-1}\,\mathbf{C}_{BA}\;}
\qquad (3\times3,\ \text{Voigt } [xx,yy,xy]). $$

Die Out-of-plane-Dehnungen relaxieren dabei frei und werden korrekt berücksichtigt — insbesondere die Querkontraktion aus der Ebene heraus und die Schubkopplung ε_yz, ε_xz, die bei beliebig orientierten Kristallen nicht verschwindet. Für einen isotropen Werkstoff reduziert sich die Formel exakt auf die bekannte Matrix E/(1−ν²)·[[1, ν, 0], [ν, 1, 0], [0, 0, (1−ν)/2]] (verifiziert, Anhang D).

**Zur Verträglichkeit der Out-of-plane-Dehnung.** Da jede Zelle ihren eigenen Tensor besitzt, kondensiert jede Zelle auch mit ihrem eigenen ε_zz (und ihren eigenen γ_yz, γ_xz). Diese Felder sind von Zelle zu Zelle unstetig und mit keinem einzelnen dreidimensionalen Verschiebungsfeld verträglich — die Lösung ist in Dickenrichtung also streng genommen inkompatibel. Aus der vorliegenden Lösung beziffert: ε_zz hat den Mittelwert −5,6·10⁻⁴ bei einer Streuung von 2,7·10⁻⁴ (27 % der aufgebrachten Dehnung ε_xx = 10⁻³), und über eine typische Korngrenze springt es um bis zu 52 % von ε_xx (99. Perzentil).

Das ist die übliche und hier akzeptierte Idealisierung: Der ebene Spannungszustand ist als dickengemittelte Aussage zu lesen. Die Verträglichkeit wird real durch einen Randschichteffekt der Größenordnung der Korngröße hergestellt, verbunden mit kleinen σ_zz im Inneren, die das Modell nicht abbildet. Für die effektive In-plane-Steifigkeit ist der Effekt zweiter Ordnung; für lokale Spannungen nahe der freien Oberfläche wäre eine echte 3D-Rechnung nötig — die aber einen zweiten Schliff oder eine Annahme über die Kornform in Dickenrichtung erfordern würde, die hier nicht vorliegt.

Der ebene Spannungszustand ist hier die physikalisch richtige Wahl: die Probe ist 2 mm dick und in Dickenrichtung frei; er ist zugleich die weichere der beiden 2D-Idealisierungen (ebene Dehnung wäre steifer). Die Aussage „das Modell ist im Übergangsbereich zu weich" wird dadurch nicht künstlich erzeugt, sondern konservativ geprüft.

## 7.6  Werkstoffzuordnung und Vorfaktor s(x)

Die Zuordnung (Tabelle\ \ref{tbl:zones}) erfolgt zweistufig: der Zonenindex bestimmt den Werkstoff, die im EBSD gemessene Phase des Korns bestimmt innerhalb dieses Werkstoffs das Kristallsystem:

| Zone | Werkstoff | Kristallsysteme im ROI (Flächenanteil) |
|---|---|---|
| 0 | 17-4PH | BCC 0,673 / FCC 0,327 |
| 1 | trans (Übergang) | FCC 0,904 / BCC 0,096 |
| 2 | 316L | FCC 1,000 |

: Werkstoff- und Phasenzuordnung im ROI. Der hohe FCC-Anteil im Übergangsbereich ist ein Messergebnis des Scans, keine Modellannahme.\label{tbl:zones}

Damit lautet die vollständige Vorschrift für die Steifigkeit einer Zelle:

$$ \mathbf{C}_{\text{Zelle}}(x) \;=\; s(x)\;\cdot\;\mathcal{P}\Big(\mathbf{M}(g_{\text{Korn}})\;\mathbf{C}^{\text{kub}}\big[\text{Werkstoff},\ \text{Kristallsystem}\big]\;\mathbf{M}(g_{\text{Korn}})^{\mathsf T}\Big) $$

mit $\mathcal{P}$ der Kondensation aus 6.5. Der Übergangsbereich besitzt in der Konfiguration einen eigenen Konstantensatz (`"trans"`), voreingestellt auf die Werte der Ausgangswerkstoffe, da seine Zusammensetzung unbekannt ist.

Der skalare Vorfaktor s(x) wirkt ausschließlich in Zone 1 und skaliert dort alle Tensorkomponenten gleichmäßig — er verändert also den Betrag der Steifigkeit, nicht die Anisotropierichtung. Voreinstellung s ≡ 1; s ist als beliebiger Ausdruck in der Lastachsenkoordinate x [µm] vorgebbar, ausgewertet am Zellmittelpunkt. Er ist die Stellschraube für Fragestellung 2 aus Kapitel 1 und erlaubt insbesondere ein glattes Profil statt eines Sprungs an den Zonengrenzen.

## 7.7  Randwertproblem und Auswertung

Gelöst wird die lineare Elastostatik im ebenen Spannungszustand,

$$ \operatorname{div}\boldsymbol\sigma = \mathbf{0},\qquad
\boldsymbol\sigma = \mathbf{C}_{\text{Zelle}}(x,y):\boldsymbol\varepsilon(\mathbf{u}),\qquad
\boldsymbol\varepsilon = \tfrac12\!\left(\nabla\mathbf{u}+\nabla\mathbf{u}^{\mathsf T}\right), $$

mit verschiebungsgesteuertem einaxialem Zug entlang der Lastachse x (Tabelle\ \ref{tbl:bcs}):

| Rand | Bedingung |
|---|---|
| x = 0 | u_x = 0 |
| x = L_x | u_x = ε₀·L_x, ε₀ = 10⁻³ |
| Eckknoten (0, 0) | u_y = 0 (Starrkörperdrehung/-verschiebung) |
| y = 0 und y = L_y | spannungsfrei (natürliche Randbedingung) |

: Randbedingungen des numerischen Zugversuchs. Oben und unten bleiben frei, die Querkontraktion ist also unbehindert.\label{tbl:bcs}

Ansatz: Q1-Verschiebungen (bilinear), Steifigkeit als DG0-Feld (elementweise konstant). Da linear elastisch gerechnet wird, ist ε₀ beliebig; alle Moduln sind davon unabhängig.

Aus der Lösung werden zwei Kennzahlen gebildet:

$$ E_{\text{Zone}} = \frac{\langle\sigma_{xx}\rangle_{\text{Zone}}}{\langle\varepsilon_{xx}\rangle_{\text{Zone}}},
\qquad
E(x_k) = \frac{\sum_{\text{Zellen} \in \text{Streifen } k}\sigma_{xx}}{\sum_{\text{Zellen} \in \text{Streifen } k}\varepsilon_{xx}} $$

mit Streifen von ≈ 100 µm Breite entlang x. Diese Definition bildet die experimentelle DIC-Auswertung nach: ⟨σ_xx⟩ ist wegen des Gleichgewichts in Lastrichtung (Serienschaltung) in allen Streifen praktisch identisch, die gesamte Variation von E(x) steckt also in der lokalen Dehnung — genau der Größe, die DIC misst. Weiche Bereiche dehnen sich stärker. Anhang B rechnet einen Datenpunkt vollständig vor.

## 7.8  Koordinatenrahmen und Konventionen

Der TSL-Kartenrahmen hat y nach unten (Bildzeilen) und z in die Ebene hinein; das FE-Modell rechnet mit y nach oben und z aus der Ebene heraus. Beide sind rechtshändig; die Umrechnung ist eine 180°-Drehung um x,

$$ \mathbf{R}_{x,180^\circ} = \operatorname{diag}(1,-1,-1), \qquad
g_{\text{FE}} = g_{\text{Karte}}\cdot \mathbf{R}_{x,180^\circ}^{\mathsf T}, $$

die konsistent auf alle Orientierungen angewandt wird (nicht durch Vorzeichenwechsel einzelner Eulerwinkel ersetzen). Zeile 0 des Mikrostrukturgitters entspricht dem oberen Kartenrand, also y_FE = L_y.

# 8  Reproduzierbarkeit

Vorverarbeitung (Host, nur numpy/scipy/PIL): `python3 preprocess_ebsd_to_grid.py --txt … --bmp … --roi 30 1339 2811 2225 --step 3.371 --zones 666.6 1494.5 --tag roi` → `micro_roi.npz` samt Vorschaubildern.

FE-Rechnung (dolfinx v0.7.3, im Container `alex-dolfinx`): `bash run_fem.sh` rechnet die drei Standardfälle (s ≡ 1, s = 1,33, Gauß-Profil) und schreibt je Fall `E_<tag>.json` (Moduln, E(x)-Profil, verwendete Konstanten), `fields_<tag>.npz` (Zellfelder auf dem Mikrostrukturgitter) und `ps_<tag>.xdmf` für ParaView. Im XDMF machen die Zellfelder `grain_id` und `E_x_local_GPa` die Kornstruktur und die kornweise Steifigkeit sichtbar; `phase_fcc1_bcc2` ist nur das Kristallsystem und hat daher zwei Werte.

Höhenstudie (Kapitel 3): `python3 study_rois.py --txt … --bmp …` schneidet aus einer Pixel→Korn-Rekonstruktion der ganzen Karte die sechs Auswertefenster aus (`micro_band1…4.npz`, `micro_full.npz`, `study_cases.json`; Zwischenspeicher `_fullmap_assign.npz`). `python3 study_bounds.py` berechnet daraus Schranken, Schätzer und Streifenprofile (`study_stats.json`) — ohne Solver. `bash run_study.sh` rechnet dieselben Fälle im dolfinx-Container (`NP=8 CASES=full bash run_study.sh` für das große Fenster). `python3 report/make_study_figs.py --src …` erzeugt die drei Abbildungen des Kapitels und ergänzt automatisch die FE-Kurven, sobald `E_<tag>.json` vorliegt.

Auswertung: `python3 make_figures.py` (Ergebnisabbildung aus den dolfinx-Dateien), `python3 report/make_report_figs.py --bmp … --src …` (Abbildungen dieses Berichts, die nur von Eingangsdaten und Materialzuordnung abhängen). Verifikation: `python3 selftest.py`, `python3 reference_solver_numpy.py …` (schreibt nach `verification/`) und `python3 compare_check.py roi`. Die Materialkonstanten stehen ausschließlich in `config.json`; `python3 materials_2d.py` zeigt die aktuell verwendeten Werte. Dieser Bericht wird gebaut mit `bash report/build_report.sh` (pandoc + xelatex); Abbildungen und Tabellen werden über `\ref`/`\label` automatisch nummeriert.

# Anhang A  Rotation des Steifigkeitstensors in Voigt-Notation

Hier wird begründet, warum die in Abschnitt 7.4 verwendete Rotationsvorschrift $\mathbf{C}' = \mathbf{M}_\sigma\mathbf{C}\mathbf{M}_\sigma^{\mathsf T}$ gilt, obwohl $\mathbf{M}_\sigma$ nicht orthogonal ist. Voigt-Vektoren werden im Folgenden durch einen Unterstrich gekennzeichnet ($\underline\sigma$, $\underline\varepsilon$), Tensoren durch Fettdruck ($\boldsymbol\sigma$, $\boldsymbol\varepsilon$).

**1. Auf Tensorebene ist nichts Besonderes.** Spannung und Verzerrung sind beide symmetrische Tensoren zweiter Stufe und transformieren sich bei einer Drehung $\mathbf{Q}$ gleich,

$$ \boldsymbol\sigma' = \mathbf{Q}\,\boldsymbol\sigma\,\mathbf{Q}^{\mathsf T},
\qquad
\boldsymbol\varepsilon' = \mathbf{Q}\,\boldsymbol\varepsilon\,\mathbf{Q}^{\mathsf T}. $$

Der Unterschied entsteht erst durch die Voigt-Darstellung: der Spannungsvektor enthält die Schubspannungen unverändert, der Verzerrungsvektor dagegen die *technischen* Gleitungen $\gamma_{ij} = 2\varepsilon_{ij}$,

$$ \underline\sigma = \big(\sigma_{xx},\,\sigma_{yy},\,\sigma_{zz},\,\sigma_{yz},\,\sigma_{xz},\,\sigma_{xy}\big)^{\mathsf T},
\qquad
\underline\varepsilon = \big(\varepsilon_{xx},\,\varepsilon_{yy},\,\varepsilon_{zz},\,2\varepsilon_{yz},\,2\varepsilon_{xz},\,2\varepsilon_{xy}\big)^{\mathsf T}. $$

Weil beide Vektoren also verschiedene Darstellungen desselben Objekttyps sind, sind auch die zugehörigen 6 × 6-Matrizen verschieden:

$$ \underline\sigma' = \mathbf{M}_\sigma\,\underline\sigma,
\qquad
\underline\varepsilon' = \mathbf{M}_\varepsilon\,\underline\varepsilon,
\qquad \mathbf{M}_\sigma \neq \mathbf{M}_\varepsilon . $$

**2. Die Dualität folgt aus der Energiedichte.** Die Formänderungsarbeitsdichte $\boldsymbol\sigma\!:\!\boldsymbol\varepsilon$ ist ein Skalar und bleibt bei einer reinen Koordinatendrehung unverändert. Die Tensorverjüngung enthält die Schubterme doppelt,

$$ \boldsymbol\sigma\!:\!\boldsymbol\varepsilon
= \sigma_{xx}\varepsilon_{xx}+\sigma_{yy}\varepsilon_{yy}+\sigma_{zz}\varepsilon_{zz}
+2\sigma_{yz}\varepsilon_{yz}+2\sigma_{xz}\varepsilon_{xz}+2\sigma_{xy}\varepsilon_{xy}, $$

und genau diese Faktoren 2 stecken in den technischen Gleitungen. Deshalb ist die Arbeitsdichte in Voigt-Notation das gewöhnliche Skalarprodukt der beiden Vektoren,

$$ \boldsymbol\sigma\!:\!\boldsymbol\varepsilon \;=\; \underline\sigma^{\mathsf T}\underline\varepsilon ; $$

das ist der eigentliche Zweck der Konvention — die 2 wandert vollständig in den Verzerrungsvektor. Aus der Invarianz $\underline\sigma^{\mathsf T}\underline\varepsilon = \underline\sigma'^{\mathsf T}\underline\varepsilon'$ wird mit den beiden Transformationen

$$ \underline\sigma^{\mathsf T}\underline\varepsilon
= \big(\mathbf{M}_\sigma\underline\sigma\big)^{\mathsf T}\big(\mathbf{M}_\varepsilon\underline\varepsilon\big)
= \underline\sigma^{\mathsf T}\,\mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\varepsilon\,\underline\varepsilon , $$

und weil das für beliebige $\underline\sigma$ und $\underline\varepsilon$ gelten muss, folgt unmittelbar

$$ \mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\varepsilon = \mathbf{I}
\qquad\Longrightarrow\qquad
\mathbf{M}_\varepsilon = \big(\mathbf{M}_\sigma^{\mathsf T}\big)^{-1}
\qquad\Longrightarrow\qquad
\mathbf{M}_\varepsilon^{-1} = \mathbf{M}_\sigma^{\mathsf T} . $$

**3. Und deshalb funktioniert die Rotation von C.** Aus $\underline\sigma = \mathbf{C}\,\underline\varepsilon$ wird nach der Drehung $\underline\sigma' = \mathbf{M}_\sigma\underline\sigma = \mathbf{M}_\sigma\mathbf{C}\,\underline\varepsilon$, und mit $\underline\varepsilon = \mathbf{M}_\varepsilon^{-1}\underline\varepsilon'$ zunächst ganz allgemein

$$ \mathbf{C}' = \mathbf{M}_\sigma\,\mathbf{C}\,\mathbf{M}_\varepsilon^{-1} . $$

Erst die eben hergeleitete Dualität macht daraus die in Abschnitt 7.4 verwendete Form

$$ \mathbf{C}' = \mathbf{M}_\sigma\,\mathbf{C}\,\mathbf{M}_\sigma^{\mathsf T} . $$

**4. Merksatz — was das bedeutet und was nicht.** Der Grund für die Transponierte ist nicht Orthogonalität ($\mathbf{M}^{-1} = \mathbf{M}^{\mathsf T}$), sondern die Energieinvarianz $\underline\sigma^{\mathsf T}\underline\varepsilon = \underline\sigma'^{\mathsf T}\underline\varepsilon' \Rightarrow \mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\varepsilon = \mathbf{I}$. Es sind zwei verschiedene Matrizen: $\mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\sigma \neq \mathbf{I}$, denn $\mathbf{M}_\sigma$ ist in der gewöhnlichen Voigt-Darstellung nicht orthogonal ($\max|\mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\sigma-\mathbf{I}| = 0{,}71$ für eine typische Drehung). Erhalten bleibt allein die *Paarung* zwischen Spannungs- und Verzerrungsvektor, nicht die Länge eines einzelnen Vektors: Spannung und Verzerrung liegen in Voigt-Notation in zueinander dualen Räumen (der eine trägt die Faktoren 2, der andere nicht), und ein dualer Vektor transformiert sich stets mit der invers-transponierten Matrix des primalen. Konkret gilt $\mathbf{M}_\varepsilon = \mathbf{D}^{2}\mathbf{M}_\sigma\mathbf{D}^{-2}$ mit der Metrik $\mathbf{D}^{2} = \operatorname{diag}(1,1,1,2,2,2)$.

Spiegelbildlich transformiert sich die Nachgiebigkeit mit der anderen Matrix, $\mathbf{S}' = \mathbf{M}_\varepsilon\mathbf{S}\mathbf{M}_\varepsilon^{\mathsf T}$ — die beiden Formen sind also nicht austauschbar.

**5. Der Sonderfall Mandel.** Skaliert man die Schubkomponenten beider Vektoren mit $\sqrt2$, verteilt sich der Faktor 2 symmetrisch auf beide Seiten. Dann fallen die beiden Transformationsmatrizen zusammen, $\mathbf{M}^{\text{Mandel}}_\sigma = \mathbf{M}^{\text{Mandel}}_\varepsilon$, und die Dualität wird zur echten Orthogonalität $\mathbf{M}^{\mathsf T}\mathbf{M} = \mathbf{I}$. Deshalb sieht die Rotationsformel dort formal gleich aus, obwohl sie eine andere Aussage macht. Beide Wege liefern denselben rotierten Tensor.

**6. Maschinelle Prüfung.** Alle Aussagen sind in `selftest.py` (Tests 9–16) geprüft: die Bond-Rotation stimmt über 50 zufällige Orientierungen bis auf $2\cdot10^{-13}$ GPa mit der exakten Rotation des Tensors vierter Stufe $C'_{ijkl} = a_{ip}a_{jq}a_{kr}a_{ls}C_{pqrs}$ überein; $\mathbf{M}_\sigma^{\mathsf T}\mathbf{M}_\varepsilon = \mathbf{I}$ gilt auf Maschinengenauigkeit; $\mathbf{M}_\sigma$ ist nachweislich nicht orthogonal; und der Umweg über Mandel liefert dasselbe Ergebnis. Der Vergleich mit der Tensorrotation vierter Stufe ist dabei das eigentliche Argument — er ist notationsunabhängig.

# Anhang B  Wie ein einzelner Datenpunkt des E(x)-Verlaufs entsteht

Der Verlauf E(x) aus Abschnitt 2.4 ist die eigentliche Vergleichsgröße zur DIC-Messung. Hier wird Schritt für Schritt beschrieben, wie ein einzelner Datenpunkt zustande kommt.

**Schritt 1 — Feldlösung.** Das Randwertproblem aus Abschnitt 7.7 wird einmal gelöst. Ergebnis ist das Verschiebungsfeld u über den gesamten ROI. Es gibt keine separate Rechnung je Streifen oder je Zone: alle Datenpunkte stammen aus derselben einen Lösung.

**Schritt 2 — Zellwerte.** Für jedes der 216 975 Elemente werden aus u zwei Zahlen ausgewertet, beide elementweise konstant (DG0-Projektion, weil auch die Steifigkeit elementweise konstant ist):

$$ \varepsilon_{xx}^{(e)} = \frac{\partial u_x}{\partial x}\bigg|_{e},
\qquad
\sigma_{xx}^{(e)} = \Big[\mathbf{C}^{\text{PS}}_{(e)}\,\underline\varepsilon^{(e)}\Big]_{xx} . $$

Man beachte: $\sigma_{xx}$ hängt über den vollen 3 × 3-Tensor auch von $\varepsilon_{yy}$ und $\gamma_{xy}$ ab — die Zelle „kennt" also ihre Nachbarschaft über die Lösung, nicht nur ihre eigene Dehnung in Lastrichtung.

**Schritt 3 — Streifenbildung.** Der ROI wird entlang der Lastachse in Streifen konstanter Breite zerlegt. Bei L_x = 2781,1 µm und einer Zielbreite von 100 µm ergeben sich 28 Streifen zu je 99,32 µm. Jeder Streifen ist 29 oder 30 Zellspalten breit und reicht über die volle Höhe des ROI (263 Zellen), enthält also rund 7600 Elemente. Ein Streifen ist damit ein 99 µm × 886 µm großes Rechteck quer zur Lastrichtung. Abb.\ \ref{fig:strips} skizziert diese Aufteilung maßstäblich.

![Skizze der Streifenaufteilung. (a) Der ROI, maßstäblich, mit allen 28 Streifen; hervorgehoben der Streifen k = 12 aus dem Rechenbeispiel (Tabelle\ \ref{tbl:example}). Die Zonengrenzen fallen nicht mit Streifengrenzen zusammen. (b) Ein Streifen von innen: 29 Spalten × 263 Zeilen = 7627 Elemente, jedes mit einem eigenen Wertepaar aus Spannung und Dehnung. (c) Die Auswertevorschrift.](fig_strips.png){#fig:strips width=15cm}

**Schritt 4 — der Datenpunkt.** Für Streifen k lautet der Wert

$$ E(x_k) \;=\; \frac{\sum_{e \in k}\sigma_{xx}^{(e)}}{\sum_{e \in k}\varepsilon_{xx}^{(e)}}
\;=\; \frac{\langle\sigma_{xx}\rangle_k}{\langle\varepsilon_{xx}\rangle_k}, $$

aufgetragen an der Streifenmitte x_k. Da alle Elemente gleich groß sind, kürzt sich die Elementzahl heraus — Summen- und Mittelwertform sind identisch.

**Warum das Verhältnis der Summen und nicht der Mittelwert der Einzelverhältnisse?** Ein zellweises $\sigma^{(e)}/\varepsilon^{(e)}$ wäre numerisch instabil (einzelne Zellen können nahezu dehnungsfrei sein) und physikalisch nicht das, was gemessen wird. Das Summenverhältnis ist dagegen exakt die Größe, die ein über den Streifen gelegtes Extensometer liefert: Kraft pro Fläche geteilt durch die mittlere Dehnung des Messfelds — also die direkte Entsprechung zur DIC-Auswertung.

**Rechenbeispiel.** Für den in Abb.\ \ref{fig:fefields} und Abb.\ \ref{fig:eprofile} rot markierten Streifen k = 12 (x = 1192 … 1291 µm, mitten in der Übergangszone):

| Größe | Wert |
|---|---|
| Zellen im Streifen | 7627 (29 Spalten × 263 Zeilen) |
| $\sum\sigma_{xx}$ | 1394,71 GPa |
| $\sum\varepsilon_{xx}$ | 7,2024 |
| $E(x_k) = \sum\sigma/\sum\varepsilon$ | 193,65 GPa |

: Rechenbeispiel: Entstehung eines einzelnen Datenpunkts des E(x)-Verlaufs aus den Zellwerten der FE-Lösung.\label{tbl:example}

**Verhältnis zu den Zonenwerten.** Die Zonenwerte aus Tabelle\ \ref{tbl:fe} entstehen nach derselben Formel, nur ist das Summationsgebiet nicht ein 99 µm breiter Streifen, sondern die gesamte Zone (637, 828 bzw. 1318 µm breit). Ein Zonenwert ist damit kein Mittelwert der zugehörigen Streifenwerte, sondern das Verhältnis der jeweiligen Summen — das ist ein dehnungsgewichtetes und damit dem Experiment entsprechendes Mittel. Ebenso ist E_apparent des gesamten ROI dieselbe Formel über alle Zellen.

**Konsistenzprüfung der abgeleiteten s-Varianten.** Die in Abschnitt 2.4 erwähnte Ableitung $E_{s}(x_k) = s(x_k)\,E_{1}(x_k)$ lässt sich am globalen Modul prüfen: die zugehörige Serienformel

$$ E_{\text{app}} = L_x \Big/ \sum_k \frac{\Delta x_k}{E(x_k)} $$

reproduziert den FE-Wert der s = 1-Rechnung auf 7 · 10⁻¹² relativ. Die abgeleiteten Kurven sind also eine belastbare Vorhersage, ersetzen aber die Bestätigungsrechnung nicht — `bash run_fem.sh` rechnet beide Varianten, und `make_report_figs.py` ersetzt die abgeleiteten Kurven danach automatisch durch die FE-Profile.

# Anhang C  Streifenmittelung gegen Feldlösung — Herleitung und Einordnung

Ergänzung zu Abschnitt 2.5 und Tabelle\ \ref{tbl:averaging}.

**Drei mögliche Mittelungen.** Aus den kornweisen Größen lassen sich drei verschiedene Streifenwerte bilden, die alle auf dasselbe Feld aus Abb.\ \ref{fig:grainEx} zurückgehen:

$$ \underbrace{\langle E_x\rangle_k}_{\text{arithmetisch}}, \qquad
\underbrace{\Big(\big\langle 1/E_x\big\rangle_k\Big)^{-1}}_{\text{harmonisch}}, \qquad
\underbrace{\Big[\big(\langle\mathbf{C}^{\text{PS}}\rangle_k\big)^{-1}\Big]_{11}^{-1}}_{\text{Tensor-Voigt}} . $$

Die ersten beiden entsprechen den klassischen Grenzfällen in Lastrichtung: das arithmetische Mittel unterstellt in allen Körnern dieselbe Dehnung (Parallelschaltung), das harmonische dieselbe Spannung (Reihenschaltung). Die dritte Variante mittelt den vollen Tensor und ist die eigentliche Voigt-Schranke aus Abschnitt 2.2.

Zwei Identitäten sind dabei nützlich und wurden numerisch bestätigt:

- Das harmonische Mittel der kornweisen E_x ist exakt die Tensor-Reuss-Schranke. Grund: $E_x = 1/S_{11}$ je Zelle, also $\langle 1/E_x\rangle = \langle S_{11}\rangle$.
- Das arithmetische Mittel ist *nicht* die Tensor-Voigt-Schranke, sondern liegt deutlich darunter (im Beispielstreifen 184 gegenüber 218 GPa). Grund: $E_x$ des Einzelkorns erlaubt bereits die freie Querkontraktion des Korns, während die Voigt-Schranke alle Dehnungskomponenten gleichsetzt und dadurch künstlich versteift.

**Der Zusammenhang mit den Voigt-/Reuss-Schranken.** Die drei Mittelungen sind nicht irgendwelche Mittelwerte, sondern genau die Konstruktionen, die den klassischen Schranken zugrunde liegen — angewandt auf einen Streifen statt auf das ganze Gebiet:

- Die Reuss-Schranke (einheitliche Spannung) *ist* die harmonische Mittelung, hier sogar als exakte Identität $1/\langle 1/E_x\rangle = 1/\langle S_{11}\rangle$.
- Die Voigt-Schranke (einheitliche Dehnung) ist die Mittelung des Steifigkeitstensors. Das arithmetische Mittel $\langle E_x\rangle$ gehört zur selben Familie (es unterstellt ebenfalls einheitliche Dehnung), ist aber keine Schranke im strengen Sinn: es mittelt den *Richtungsmodul* statt des Tensors und lässt damit die Querkontraktion jedes Korns frei — deshalb liegt es systematisch unter der echten Voigt-Schranke.
- Die Schranken in Abschnitt 2.2 sind dieselbe Rechnung über ganze Zonen statt über 99-µm-Streifen.

Es gilt daher durchgehend die Ordnung Reuss ≤ arithmetisch ≤ Voigt (für alle 28 Streifen numerisch bestätigt). Wo diese Ordnung entartet — im einkristallinen 316L-Bereich mit nur einem Korn und damit ohne Steifigkeitskontrast — fallen alle drei zusammen.

Ein wichtiger Vorbehalt: Voigt und Reuss sind Schranken für einen Körper unter uniformen Randbedingungen. Ein Streifen ist aber kein eigenständiger Körper, sondern wird von seinen Nachbarn belastet; sein scheinbarer Modul muss deshalb nicht zwingend in seinen eigenen Schranken liegen. Empirisch tut er es hier: in allen 15 Streifen mit nennenswertem Kontrast (Schrankenabstand > 5 GPa) liegt die FE-Lösung innerhalb ihres Intervalls.

**Warum die arithmetische Mittelung so gut trifft.** Normiert man die Lage im Schrankenintervall auf 0 (Reuss) bis 1 (Voigt), liegt die FE-Lösung im Mittel bei 0,40 (Spanne 0,23 … 0,57) — und das arithmetische Mittel bei 0,39. Beide sitzen also praktisch an derselben Stelle zwischen den Schranken; das arithmetische Mittel des Richtungsmoduls ist damit ein überraschend guter *Schätzer* für die tatsächliche Lösung, nicht bloß eine Schranke.

Die Höhenstudie hat diesen Befund inzwischen auf sechs sehr verschieden aufgebaute Fenster ausgeweitet (Abschnitt 3.6): über 18 Zonenwerte liegt der Schätzer im Mittel 2,0 GPa über der FE-Lösung, maximal 7,7 GPa. Die gute Übereinstimmung ist also kein Einzelfall des Referenzbandes.

Das ist kein Zufall, aber auch keine Identität. Ein Streifen ist 99 µm breit und 886 µm hoch: quer zur Last liegen die Körner parallel (das treibt zum arithmetischen Mittel), längs zur Last liegen innerhalb des Streifens rund 29 Zellreihen hintereinander (das treibt zum harmonischen Mittel). Die tatsächliche Lösung liegt zwischen beiden, und bei diesem Seitenverhältnis überwiegt die Parallelanordnung.

Ein klarer Konsistenztest steckt im 316L-Bereich: dort besteht das Gefüge aus einem einzigen Korn, es gibt keinen Steifigkeitskontrast, und folglich fallen alle vier Kurven exakt zusammen (Abb.\ \ref{fig:averaging}b, rechter Teil: Abweichung null). Genau dort, wo Mittelung und Lösung übereinstimmen müssen, tun sie es auch — und genau dort, wo das Gefüge am heterogensten ist (Übergangszone, x ≈ 1,2 mm), wird die Abweichung am größten.

# Anhang D  Verifikation

Die Implementierung wurde unabhängig vom Ergebnis geprüft:

- **Einheitentests der Kristallmathematik** (11 Prüfungen, `selftest.py`): Rotationsinvarianz eines isotropen Tensors; kubische Invarianz unter 90°-Drehung; analytische Einkristallmoduln E⟨100⟩/E⟨111⟩ getroffen; Kondensation eines isotropen Tensors ergibt exakt E/(1−ν²)·[…]; Kondensation identisch zur vollständigen Lösung mit freien Out-of-plane-Dehnungen; Symmetrie und positive Definitheit des kondensierten Tensors; Mittelung über zufällige Orientierungen konvergiert gegen In-plane-Isotropie; Doppelanwendung der Rahmendrehung ergibt die Identität. Hinzu kommen die Tests 9–16 zur Rotationsvorschrift (Anhang A).
- **Patch-Test:** homogen-isotropes Ersatzgefüge liefert exakt E_apparent = E und ν_xy = ν; mit s ≡ 2 exakt den doppelten Wert.
- **Gleichgewichtstest:** ⟨σ_xx⟩ in allen Zonen identisch (Tabelle\ \ref{tbl:fe}).
- **Schrankentest:** alle FE-Werte liegen zwischen Reuss- und Voigt-Schranke ihrer Zone (Tabelle\ \ref{tbl:bounds}); in der Höhenstudie gilt das für alle 18 Zonenwerte aus sechs unabhängigen Rechnungen (Abschnitt 3.6).
- **Solver-Kreuzvergleich:** ein unabhängig implementierter numpy/scipy-Q1-Referenzsolver (gleiches Netz, gleiche Randbedingungen, gemeinsames Materialmodul) liefert E_apparent = 182,865200236 GPa gegenüber 182,865200238 GPa aus dolfinx — relative Abweichung 1,3 · 10⁻¹¹; die Zonenmoduln stimmen ebenso überein.
- **Netzstudie:** Halbierung der Auflösung (6,742 µm statt 3,371 µm, 53 972 statt 216 975 Zellen) ändert E_apparent um weniger als 0,5 % — die effektive Steifigkeit ist auflösungskonvergiert. (Diese Studie wurde mit dem Prüfsolver durchgeführt; eine Wiederholung mit dolfinx ist über `run_fem.sh` möglich.)

# Anhang E  Herkunft der Einkristallkonstanten

Die Werte in Tabelle\ \ref{tbl:singlecrystal} sind Literaturwerte für verwandte, nicht für die hier untersuchten Werkstoffe. Sie wurden unverändert aus der Konfiguration der Neper-Studie (Projekt 069) übernommen, damit beide Arbeiten dieselbe Basis nutzen und die Ergebnisse untereinander vergleichbar bleiben.

**FCC (Austenit).** Der Satz C11 = 204,6 / C12 = 137,7 / C44 = 126,2 GPa ist in der Konfiguration einem austenitischen Cr-Ni-Stahl vom 304-Typ nach Ledbetter zugeordnet [1]. Diese Zuordnung stammt aus der Projekthistorie; die Zahlenwerte wurden im Rahmen dieser Arbeit nicht gegen die Primärquelle geprüft (die Arbeit ist nicht frei zugänglich). Sie liegen im für austenitische Stähle üblichen Bereich, sind aber weder für 316L noch für den hier vorliegenden WAAM-Zustand gemessen.

**BCC (Martensit / Ferrit).** Der Satz C11 = 231,4 / C12 = 134,7 / C44 = 116,4 GPa entspricht reinem α-Eisen bei Raumtemperatur. Er ist mit unabhängig publizierten Messungen konsistent und liegt zwischen ihnen (Tabelle\ \ref{tbl:bccsrc}):

| Quelle | C11 | C12 | C44 |
|---|---|---|---|
| Dever (1972) [3] | 232,2 | 135,6 | 117,0 |
| Resonanz-Ultraschall (RPR) [4] | 230,5 | 133,3 | 116,3 |
| hier verwendet | 231,4 | 134,7 | 116,4 |

: Vergleich des verwendeten BCC-Satzes mit publizierten Messungen an α-Eisen bei Raumtemperatur [GPa]. Die verwendeten Werte liegen innerhalb der Messstreuung der Literatur.\label{tbl:bccsrc}

**Warum das eine Unsicherheit bleibt.** Für die hier untersuchten Werkstoffe sind beide Sätze aus drei Gründen nur Näherungen:

- *Legierungseinfluss.* Beide Werte gelten für andere Zusammensetzungen (304-Typ-Austenit bzw. reines Eisen). Insbesondere 17-4PH enthält rund 16 % Cr, 4,5 % Ni und 3,3 % Cu; Cu ist in gelöster Form und als Ausscheidung vorhanden.
- *Kristallstruktur des Martensits.* Martensit ist streng genommen kubisch-raumzentriert-tetragonal (bct), nicht kubisch. Bei niedrigem Kohlenstoffgehalt (17-4PH: 0,04 %) ist die Tetragonalität gering, sodass die kubische Näherung vertretbar ist; sie bleibt aber eine Näherung.
- *Ausscheidungszustand.* 17-4PH ist ein ausscheidungshärtender Stahl. Kupferausscheidungen verändern die effektive Steifigkeit gegenüber reinem α-Eisen; dieser Effekt ist in Tabelle\ \ref{tbl:singlecrystal} nicht enthalten und ist zugleich einer der Kandidaten für die im Übergangsbereich beobachtete Versteifung (Kapitel 4).

**Konsequenz für die Ergebnisse.** Alle Absolutwerte skalieren mit diesen Konstanten; die Voigt-Schranke aus Abschnitt 2.2 skaliert ebenfalls annähernd proportional. Die Kernaussage — der gemessene Übergangswert liegt oberhalb der Voigt-Schranke — bleibt so lange bestehen, wie die tatsächlichen Konstanten des Übergangswerkstoffs nicht um mehr als etwa 14 % über den hier angesetzten liegen. Genau das ist die zu klärende Frage (Kapitel 4); die Konstanten sind daher vor einer Publikation durch gemessene oder zumindest legierungsspezifisch validierte Werte zu ersetzen. Der Ort dafür ist ausschließlich `config.json`.

# Literatur

[1] H. M. Ledbetter: *Monocrystal–Polycrystal Elastic Constants of a Stainless Steel*. physica status solidi (a) 85 (1984) 89–96. DOI: 10.1002/pssa.2210850111 — zugehörig: H. M. Ledbetter: *Predicted monocrystal elastic constants of 304-type stainless steel*. Physica B+C 128 (1985) 1–4.

[2] J. A. Rayne, B. S. Chandrasekhar: *Elastic Constants of Iron from 4.2 to 300 K*. Physical Review 122 (1961) 1714–1716. DOI: 10.1103/PhysRev.122.1714

[3] D. J. Dever: *Temperature dependence of the elastic constants in α-iron single crystals: relationship to spin order and diffusion anomalies*. Journal of Applied Physics 43 (1972) 3293–3301.

[4] Resonanz-Ultraschall-Messungen an α-Eisen-Einkristallen bei 300 K, zusammengestellt in der Vergleichstabelle von X. Sha, R. E. Cohen: *First-principles thermoelasticity of bcc iron under pressure*. Physical Review B 74 (2006) 214111 (arXiv:cond-mat/0612308).
