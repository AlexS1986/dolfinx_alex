---
title: "Kristallachsen, Textur und Polfiguren — kurz erklärt"
subtitle: "Hintergrund zur WAAM-Anisotropie-Arbeit (316L / 17-4PH)"
lang: de
geometry: margin=2.3cm
fontsize: 11pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
header-includes: |
  \setcounter{secnumdepth}{-1}
  \usepackage{float}
  \floatplacement{figure}{H}
  \setlength{\emergencystretch}{4em}
---

# 1  Körner sind kleine Kristalle

Ein Werkstoff wie 316L besteht aus vielen *Körnern*. Jedes Korn ist ein kleiner Einkristall: Die Atome sitzen auf einem regelmäßigen Gitter. Bei 316L (Austenit) ist dieses Gitter kubisch-flächenzentriert (FCC) — ein Würfel mit Atomen an den Ecken und in den Flächenmitten (Abb.\ \ref{fig:cube}). 17-4PH (Martensit) ist kubisch-raumzentriert (BCC), aber ebenfalls kubisch.

# 2  Die ⟨100⟩-Kristallachsen

Der Würfel hat drei Kantenrichtungen: [100], [010] und [001]. Weil sie durch die Würfelsymmetrie gleichwertig sind, fasst man sie zur Richtungsfamilie ⟨100⟩ zusammen (spitze Klammern = ganze Familie, eckige Klammern = eine bestimmte Richtung). Die ⟨100⟩-Achsen sind also einfach die drei Würfelkanten-Richtungen des Kristallgitters.

Warum das mechanisch zählt: In einem kubischen Kristall hängt die Steifigkeit von der Belastungsrichtung ab. Für Austenit ist ⟨100⟩ die *nachgiebigste* Richtung (E ≈ 94 GPa), die Raumdiagonale ⟨111⟩ die *steifste* (≈ 300 GPa). Ein einzelnes Korn ist demnach je nach Richtung sehr unterschiedlich steif — genau das macht ein texturiertes Gefüge richtungsabhängig.

![Kubische Elementarzelle (FCC) mit den drei ⟨100⟩-Kantenrichtungen und der ⟨111⟩-Raumdiagonale. Für Austenit ist ⟨100⟩ weich, ⟨111⟩ steif.](expl_A_cube.png){#fig:cube width=11cm}

# 3  Orientierung und Textur

Im Gefüge ist jedes Korn anders gedreht. Die *Orientierung* eines Korns beschreibt, wie sein Würfel relativ zu den Probenachsen (Aufbau, Schweiß, Wandnormale) verkippt ist; EBSD misst sie für jedes Korn (als Eulerwinkel). Die Gesamtheit aller Orientierungen heißt *Textur*.

Sind die Körner regellos gedreht (Abb.\ \ref{fig:grains}, links), so mittelt sich die Richtungsabhängigkeit über viele Körner heraus — das Gefüge ist makroskopisch isotrop. Zeigen dagegen viele Körner mit derselben Kristallachse in dieselbe Probenrichtung (Abb.\ \ref{fig:grains}, rechts: ⟨100⟩ ∥ Aufbau und Schweiß), liegt eine Textur vor — und das Gefüge wird richtungsabhängig (anisotrop). Eine ⟨100⟩-Textur entlang der Aufbaurichtung würde diese Richtung z. B. weich machen (weil ⟨100⟩ die weiche Kristallrichtung ist).

![Ein Gefüge ist eine Ansammlung von Körnern (Kästchen) mit je eigener Gitter-Orientierung; der rote Pfeil markiert eine ⟨100⟩-Achse. Links regellos (isotrop), rechts texturiert (⟨100⟩ an den Probenachsen ausgerichtet → anisotrop).](expl_B_grains.png){#fig:grains width=15.5cm}

# 4  Was ist eine Polfigur?

Eine Polfigur ist eine Landkarte aller Kornorientierungen auf einer einzigen Scheibe. Man wählt eine Kristallrichtungsfamilie (hier ⟨100⟩) und trägt für jedes Korn ein, wohin seine ⟨100⟩-Achsen relativ zur Probe zeigen. Die Umrechnung „Richtung → Punkt auf der Scheibe" leistet die *stereografische Projektion* (Abb.\ \ref{fig:stereo}): Man blickt entlang einer festen Achse — bei unseren Schliffen ist das die Schliffnormale. Eine Kristallrichtung, die fast parallel zur Blickrichtung liegt, landet nahe der Scheibenmitte; eine Richtung, die fast in der Ebene liegt, nahe dem Rand.

![Stereografische Projektion: Eine Kristallrichtung (Pfeil) wird zu einem Punkt auf der Scheibe. Fast senkrechte Richtungen landen nahe dem Zentrum, fast waagerechte nahe dem Rand.](expl_C_stereo.png){#fig:stereo width=11.5cm}

Damit lässt sich die Textur direkt ablesen (Abb.\ \ref{fig:schem}): Bei regelloser Textur streuen die Punkte gleichmäßig über die ganze Scheibe. Bei scharfer ⟨100⟩-Textur bilden sie enge Häufungen — für ⟨100⟩ ∥ Blickrichtung eine Häufung im Zentrum und, wegen der Würfelsymmetrie (die anderen beiden ⟨100⟩ liegen dann in der Ebene, um 90° versetzt), vier weitere Häufungen am Rand. Die Punktgröße gewichtet man üblicherweise mit der Kornfläche, damit große (mechanisch wichtige) Körner stärker zählen.

![{100}-Polfigur, schematisch. Links regellose Textur (gleichmäßige Streuung), rechts scharfe ⟨100⟩-Textur (zentrale Häufung + vier Randhäufungen).](expl_D_polefig.png){#fig:schem width=14cm}

# 5  Im Kontext unserer Arbeit

In der Anisotropie-Arbeit nehmen wir EBSD-Schliffe von WAAM-316L auf (Vertikal, Horizontal, 45°) und zeichnen daraus {100}-Polfiguren (Abb.\ \ref{fig:real}). Die Zugversuche zeigen, dass 316L entlang Aufbau und Schweiß sehr weich ist (E ≈ 94 GPa) — es wird dort also fast wie ein ⟨100⟩-Einkristall belastet. Läge im Gefüge tatsächlich eine scharfe ⟨100⟩-Textur mit ⟨100⟩ ∥ Aufbau vor, müsste die Polfigur eine deutliche zentrale Häufung zeigen (wie in Abb.\ \ref{fig:schem}, rechts).

Die gemessenen Schliffe (Abb.\ \ref{fig:real}) zeigen aber nur eine breite Streuung ohne dominante Häufung — also eine nur schwach texturierte Mikrostruktur. Genau das ist der offene Punkt aus Abschnitt 3.6 des Hauptberichts: Die vorliegenden Scans tragen die scharfe ⟨100⟩-Textur nicht, die die Zugversuche nahelegen. Ob dahinter inkonsistente Referenzrahmen der getrennt gescannten Schliffe stehen (die ungeprüften EBSD-Konventionen) oder der Scanbereich wirklich schwächer texturiert ist, muss datenbasiert geklärt werden — durch Prüfung der Orientierungskonventionen, nicht dadurch, dass man dem Modell die „passende" Textur vorgibt.

![{100}-Polfiguren der drei realen 316L-Schliffe (EBSD-Rohdaten, flächengewichtet). Zentrum (+) = Schliffnormale. Die breite Streuung ohne dominante Häufung entspricht einer nur schwach texturierten Mikrostruktur — die scharfe ⟨100⟩-Textur der Zugproben ist hier nicht sichtbar.](expl_E_real.png){#fig:real width=16cm}
