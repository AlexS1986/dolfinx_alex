# Mailentwurf — Versand des Berichts

**Betreff:** WAAM N=1, Übergangsbereich: Mikrostruktur erklärt die Steifigkeitsüberhöhung nicht — Bericht

**Anhänge:** `WAAM_N1_transition_kurzfassung.pdf` (7 Seiten), `WAAM_N1_transition_report.pdf` (34 Seiten)

---

Hallo Ralf,

anbei der Bericht zur hybriden N=1-Probe. Die Kurzfassung (7 Seiten) fasst Modell und Ergebnisse zusammen und ist für sich lesbar; der ausführliche Bericht enthält zusätzlich die Herleitungen, ein Rechenbeispiel, das Verifikationsprotokoll und die Quellenlage der Konstanten.

Kurz zum Stand: Ich habe die im EBSD-Scan gemessene Mikrostruktur explizit modelliert — die tatsächliche Kornkarte, ein Q1-Element je EBSD-Pixel, jedes Korn mit seinem eigenen rotierten und auf ebenen Spannungszustand kondensierten Einkristalltensor. Das Ergebnis ist eindeutig: Die gemessene Überhöhung im Grenzflächenbereich (232,4 ± 13 GPa) lässt sich damit nicht reproduzieren. Im Modell ist der Übergangsbereich sogar die weichste Zone.

Der Punkt, der mir am wichtigsten ist, hängt dabei an keiner FE-Rechnung: Die Voigt-Schranke derselben Kornstruktur — die obere Grenze über alle denkbaren Anordnungen dieser Körner mit diesen Einkristallkonstanten — liegt 14 % unter dem Messwert. Ich habe das inzwischen an sechs Auswertefenstern über die gesamte Scanhöhe geprüft, einschließlich eines Fensters über die volle Probenhöhe mit 30 673 Körnern: In keinem einzigen erreicht die Schranke den Messwert, der Abstand liegt zwischen 10 und 50 %. Der Steifigkeitstensor des Übergangswerkstoffs selbst müsste also um den Faktor 1,33 bis 1,48 höher liegen.

Als Gegenprobe: Dasselbe Modell mit denselben Konstanten trifft über die volle Probenhöhe den Monowerkstoff 316L auf 1,6 % (160,2 gegenüber gemessenen 162,7 GPa). Im kleinen Referenzausschnitt lag es dort noch 14 % daneben — das war ein reiner Stichprobeneffekt, weil dieser Ausschnitt zu 99,8 % aus einem einzigen Chevron-Korn besteht. Für mich ist das der stärkste Hinweis darauf, dass die Diskrepanz im Werkstoff steckt und nicht im Modell.

Zwei Dinge, bei denen ich Unterstützung aus der Werkstoffkunde bräuchte:

1. Die Einkristallkonstanten sind derzeit Literaturwerte verwandter Werkstoffe (304-Typ-Austenit bzw. reines α-Eisen), nicht für 316L, 17-4PH oder den Übergangswerkstoff gemessen. Alle Absolutwerte skalieren mit ihnen, deshalb würde ich sie vor einer Veröffentlichung gern durch belastbarere Werte ersetzen.
2. Gibt es eine gefügekundliche Erklärung für eine so deutliche lokale Versteifung? Eine Spur habe ich: Das Steifigkeitsmaximum fällt mit dem steilsten Gradienten des Ferritgehalts zusammen — gemessen an derselben Probe und denselben 14 Messpunkten wie der E-Verlauf. Die Überhöhung sitzt also nicht dort, wo viel Ferrit ist, sondern dort, wo sich die Phasenzusammensetzung am schnellsten ändert. Das würde zu einem Erstarrungs- oder Aufmischungseffekt passen; eine Gefügeanalyse in genau dieser Zone (Messpunkte 4 bis 6) wäre der nächste sinnvolle Schritt.

Über Rückmeldung zu beiden Punkten würde ich mich freuen — insbesondere, ob der Faktor von rund einem Drittel materialkundlich überhaupt plausibel zu machen ist. Für eine Besprechung stehe ich gern zur Verfügung.

Viele Grüße
Alexander
