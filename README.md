# 🧩 Prognosekombination – viele Prognosen zu einer

Neuntes Stück der **Zeitreihen-Prognose-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Der **Zusammenfluss** der Verfahren aus den Stücken 1, 2, 4 und 6 ([Naive Prognose](https://github.com/sebastian-hanisch/naive-forecast-demo), [Exponentielle Glättung](https://github.com/sebastian-hanisch/exponential-smoothing-demo),
[Dynamische Regression](https://github.com/sebastian-hanisch/dynamic-regression-demo), [Boosting](https://github.com/sebastian-hanisch/boosting-forecast-demo)). Geplant sind zwei weitere Stücke (Prognose → Bestand, ein vortrainiertes Netz; noch nicht gebaut).

Wer eine Nachfrage prognostiziert, hat meist mehrere Modelle zur Hand. Welches ist das beste? Das weiß man erst **hinterher**. Die Alternative: **alle verwenden** und ihre Prognosen **kombinieren** – als einfachen Mittelwert, als Median, mit Gewichten nach vergangener Güte oder mit Gewichten, die man aus den letzten Fehlern schätzt.
Die Demo nimmt sechs Verfahren als **Pool** und misst auf einem **Portfolio von Depots**, **wann die Kombination** dem besten Mitglied nahe kommt oder es schlägt, **wann der einfache Mittelwert genügt** und was ein schlechtes Mitglied im Pool anrichtet. Alle Daten sind erzeugt, die Rechnung ist in numpy geschrieben (`scipy` nur als Gegenprobe im Test).

**Bezug zu OR:** Prognosen sind Eingaben von Planungsmodellen; ein robuster Verbund ist oft wertvoller als das zufällig beste Einzelmodell, zumal man das beste nur nachträglich kennt.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie erwartete: "Der einfache Mittelwert ist kaum zu schlagen; die Kombination ist besser als jedes Mitglied." Die Messung sagt: **Die Kombination gewinnt wenig gegenüber dem besten Mitglied (typisch 1 bis 4 %), aber sie muss es vorher nicht kennen – und der einfache Mittelwert ist keineswegs der beste Kombinierer.**

1. **Das beste Mitglied wechselt mit den Umständen, die Kombination folgt ihm.** Ohne Niveauschwankung ist die Regression mit 0,744 das beste Mitglied (sie kennt die wahre Form des Vehikels); bei Schwankung 0,12 fällt sie auf 1,456, das Boosting (0,890) ist vorn. Die adaptiven Gewichte liegen in beiden Fällen bei oder unter dem besten Mitglied (0,743 und 0,871), die **Auswahl nach dem Fenster** nicht (0,747 und 0,922).
2. **Der einfache Mittelwert ist robust, aber nicht der Beste.** Im Standardfall (Wochenmittel, Holt-Winters, Regression, Boosting, Seed 3) liegt er mit 0,842 um 1,0 % über dem besten Mitglied (Boosting, 0,833); Inverse MSE −0,8 %, Kleinste Quadrate (nicht negativ) −1,1 %, **adaptiv 0,815 (−2,2 %)**. Ohne Schwankung kommt der Mittelwert dem besten Mitglied nahe, wird aber nicht besser (+0,7 %); die Kleinsten Quadrate gewinnen −4,0 %.
3. **Ein schlechtes Mitglied bricht den Mittelwert, nicht die Gewichte.** Mit dem Mitglied Naiv (MASE 2,853) im Pool steigt der Mittelwert auf 1,045 (+25,5 % gegen das beste Mitglied); Median (+2,4 %) und getrimmter Mittelwert (+2,1 %) verkraften es, geschätzte Gewichte kaum spürbar (adaptiv −2,3 %, Kleinste Quadrate nicht negativ −1,0 %).
4. **Gewichte zu schätzen kostet Schätzfehler.** Mit 30 Ursprüngen im Kalibrierfenster erreichen die **freien** Kleinsten Quadrate 0,850 (bei 120 Ursprüngen 0,805); die auf nicht negative Gewichte mit Summe 1 beschränkte Fassung bleibt bei 0,812 gegen 0,806. Der Mittelwert schätzt nichts und liegt konstant bei 0,826. Das ist das **Kombinations-Rätsel** der Literatur.
5. **Auswahl ist keine Kombination.** Das Mitglied mit dem kleinsten Fehler im Fenster zu nehmen, liegt im Standardfall bei +2,5 % gegen das beste Mitglied im Nachhinein – schlechter als jeder Kombinierer, weil das Fenster verrauscht ist und das beste Mitglied von Depot zu Depot wechselt (Boosting in 22 von 30 Depots, Regression in 8).
6. **Ergänzung schlägt Güte.** Ein Pool aus nur Holt-Winters (0,887) und Regression (1,148) – der schwächere Partner ist deutlich schlechter – ergibt mit freien Kleinsten Quadraten 0,834 (−5,9 % gegen das beste Mitglied), weil die beiden in verschiedenen Depots gut sind (Holt-Winters in 18, Regression in 12); der Mittelwert liegt bei +1,2 %.

## Modell

- **Das Portfolio** (`cmb_scenario.py`): 1 095 Tage je Depot wie in den Vorgängern (multiplikativ: Niveau, Trend, Wochen- und Jahresmuster, Feiertage, Aktionen, log-normales Rauschen, eigene Parameter je Depot); neu ist eine **depoteigene Niveauschwankung** (mittelwertrückkehrender Zufallsgang im Log, $\varphi = 0{,}98$, je Depot mit eigenem Faktor um den Reglerwert): sie macht Depots verschieden, in denen die statische Regression nicht mehr folgen kann.
- **Der Pool** (`cmb_members.py`): **Naiv** (letzter Wert), **Letzte Woche**, **Wochenmittel** (vier Wochen, Stück 1), **Holt-Winters** multiplikativ (Stück 2), **Regression** im Log auf Kalender und Aktionsplan (Stück 4) und das **globale Boosting** über alle Depots (Stück 6; Zieltage vor Tag 610, Ziel = log-Verhältnis zum 28-Tage-Niveau). Die Parameter stammen aus den Tagen vor Tag 610; die Tage 610–729 sind das **Kalibrierfenster**, das Testjahr beginnt bei Tag 730.
- **Kombinationen** (`cmb_combine.py`): Mittelwert, Median, getrimmter Mittelwert (ohne das kleinste und größte Mitglied, ab vier Mitgliedern), **Inverse MSE** (Bates/Granger), **Kleinste Quadrate frei** (Granger/Ramanathan), **Kleinste Quadrate nicht negativ mit Summe 1** (exakt: alle Träger durchprobiert), **adaptiv** ($w_k \propto D_k^{-\eta}$ mit exponentiell abklingendem Fehler; $\eta = 0$: Mittelwert, $\eta = 1$: Inverse MSE, groß: Auswahl) und **Auswahl** des Mitglieds mit dem kleinsten Fehler.
  Gewichte werden **nur aus realisierten Fehlern** geschätzt: zu einem Testursprung $t$ zählen die Ursprünge $o \le t - h$ der letzten $W$; Ist und Prognosen sind durch die mittlere Mitglieder-Prognose geteilt (relative Fehler).
- **Kennzahl** (`cmb_evaluation.py`): **MASE** je Depot (MAE über alle Testursprünge und Horizonte durch den saisonal naiven Trainingsfehler), über die Depots gemittelt. Referenzen: das beste Mitglied **im Nachhinein** und das beste Mitglied **je Depot im Nachhinein** (beide erst nach dem Testjahr bekannt, für keine Methode erreichbar).

## Methodik

- **Handrechnungen:** Mittelwert, Median und getrimmter Mittelwert von fünf Zahlen (10, 20, 30, 40, 100 → 40, 30, 30); Inverse MSE und Auswahl auf zwei Mitgliedern (Fehler 1 und 2 → MSE 1 : 4 → Gewichte 0,8 : 0,2); der Simplex-Löser auf $A = I$ (Projektion von (0,7; 0,2; −0,3) auf das Simplex → (0,75; 0,25; 0)); die Fensterregel; die Mitglieder Naiv, Letzte Woche und Wochenmittel; MASE und Referenzen.
- **Gegenprobe:** der Simplex-Löser gegen `scipy.optimize.minimize` (SLSQP) samt KKT-Bedingungen; die freien Kleinsten Quadrate gegen `numpy.linalg.lstsq`; die Fenstersummen und das adaptive Verfahren gegen **unabhängige Schleifen**; die wahren Gewichte $(0{,}7;\ 0{,}3)$ werden aus $y = 0{,}7 F_1 + 0{,}3 F_2$ zurückgewonnen.
- **Eigenschaften:** die Gewichte von Mittelwert, Inverse MSE, Simplex, adaptiv und Auswahl sind Verteilungen; $\eta = 0$ ist der Mittelwert; **die Prognosen bis zu einem Ursprung ändern sich nicht, wenn Ist-Werte ab dem Ursprung überschrieben werden** (Mitglieder und Kombinationen), die Mitglieder schätzen ihre Parameter nur aus den Tagen vor Tag 610; das Boosting ist deterministisch.
- **Eine Korrektur unterwegs:** das Simplex-Problem war zuerst mit einem beschleunigten Gradientenverfahren gelöst (Genauigkeit 0,3 % bei schlecht konditionierten Mitgliedern); jetzt exakt über alle Träger (bei sechs Mitgliedern 63 Gleichungssysteme je Depot und Ursprung, vektorisiert). Die Ergebnisse änderten sich in der dritten Stelle nicht.
- **Statistik:** die Experimente mitteln über **drei feste Seeds** (Fehlerbalken = Standardfehler), die Preset-Zeilen sind **Einzelportfolios** (Seed 3).
- **Literatur** (nicht nachgebaut): Bates/Granger 1969 (Kombination von Prognosen); Granger/Ramanathan 1984 (Kleinste-Quadrate-Gewichte); Clemen 1989 (Übersicht); Timmermann 2006 (Kombination, das Rätsel des einfachen Mittelwerts); Makridakis et al. (M4/M5: Kombinationen an der Spitze); Hyndman/Athanasopoulos, FPP3.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Preset, vier Mitglieder, 30 Depots, Schwankung 0,06) | MASE: Wochenmittel 0,979, Holt-Winters 0,887, Regression 1,148, **Boosting 0,833** (bestes Mitglied im Nachhinein; in 22 von 30 Depots das beste, Regression in 8). Kombinationen gegen das beste Mitglied: Mittelwert 0,842 (+1,0 %), Median +0,6 %, Inverse MSE −0,8 %, Kleinste Quadrate (nicht negativ) −1,1 %, **adaptiv 0,815 (−2,2 %)**, Auswahl 0,853 (+2,5 %). Bestes Mitglied je Depot im Nachhinein 0,822. | `test_standard_preset` |
| **Ohne Schwankung** (Preset) | Regression 0,775 (bestes Mitglied), Boosting 0,790; Kleinste Quadrate frei 0,744 (−4,0 %), adaptiv 0,745 (−3,8 %), Auswahl 0,758 (−2,1 %), Mittelwert 0,780 (+0,7 %). | `test_no_swing_preset` |
| **Starke Schwankung** (Preset, 0,12) | Regression 1,712, Boosting 0,906 (bestes Mitglied); Mittelwert 0,953 (+5,2 %), Kleinste Quadrate frei 0,950 (+4,9 %), Median 0,908 (+0,2 %), adaptiv 0,894 (−1,4 %). | `test_strong_swing_preset` |
| **Ein schlechtes Mitglied** (Preset, Naiv im Pool) | Naiv 2,853; Mittelwert 1,045 (+25,5 %), Median 0,853 (+2,4 %), getrimmt 0,850 (+2,1 %), Kleinste Quadrate (nicht negativ) 0,825 (−1,0 %), adaptiv 0,814 (−2,3 %). | `test_bad_member_preset` |
| **Kleines Fenster** (Preset, 30 Ursprünge) | Kleinste Quadrate frei 0,865 (+3,9 %), nicht negativ 0,827 (−0,7 %), adaptiv 0,817 (−1,9 %), Mittelwert 0,842 (+1,0 %). | `test_small_window_preset` |
| **Zwei Mitglieder** (Preset, Holt-Winters und Regression) | Holt-Winters 0,887, Regression 1,148; Mittelwert 0,897 (+1,2 %), Kleinste Quadrate frei 0,834 (−5,9 %), adaptiv 0,850 (−4,1 %), Inverse MSE 0,852 (−3,9 %); Holt-Winters ist in 18 Depots besser, die Regression in 12. | `test_two_member_preset` |
| **Niveauschwankung** (Experiment, 20 Depots, 3 Seeds; Schwankung 0 / 0,06 / 0,12) | Regression 0,744 / 1,011 / **1,456**; Boosting 0,804 / 0,835 / 0,890; Mittelwert 0,780 / 0,826 / 0,915; Median 0,781 / 0,826 / 0,884; Kleinste Quadrate frei 0,739 / 0,811 / 0,913; Simplex 0,739 / 0,806 / 0,879; adaptiv 0,743 / 0,804 / **0,871**; Auswahl 0,747 / 0,850 / 0,922; bestes Mitglied je Depot im Nachhinein 0,742 / 0,820 / 0,881. | `test_swing_experiment` |
| **Pool wächst** (Experiment; 2 / 3 / 4 / 5 / 6 Mitglieder, vom besten zum schlechtesten) | Bestes Mitglied 0,835. Mittelwert 0,843 / 0,816 / 0,826 / 0,850 / **0,982**; adaptiv 0,804 / 0,799 / 0,804 / 0,805 / 0,804; Simplex 0,808 / 0,805 / 0,806 / 0,807 / 0,807; Kleinste Quadrate frei 0,805 (zwei Mitglieder) und 0,815 (sechs); Median bei sechs Mitgliedern 0,855, Inverse MSE 0,828, Auswahl 0,848. | `test_pool_experiment` |
| **Kalibrierfenster** (Experiment, 30 / 60 / 120 Ursprünge) | Kleinste Quadrate frei **0,850** / 0,822 / 0,805; Simplex 0,812 / 0,807 / 0,806; adaptiv 0,807 / 0,805 / 0,803; Inverse MSE 0,811 (bei 30), Auswahl 0,845, Mittelwert konstant 0,826. | `test_window_experiment` |

Die Preset-Zeilen sind **Einzelportfolios** (Seed 3); belastbar sind die Zeilen über drei Seeds.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Mitglieder sind ähnlich gut** | Ist eines viel schlechter, zieht es den einfachen Mittelwert nach unten (+25 % im Preset); Median, getrimmter Mittelwert und geschätzte Gewichte sind robuster. | Pool vorher prüfen, Gewichte schätzen |
| **Gewichte lassen sich schätzen** | Aus kurzen Fenstern sind sie verrauscht, besonders die freien (0,850 statt 0,805); die Beschränkung hilft. | Beschränkung, größeres Fenster, Schrumpfung zum Mittelwert |
| **Die Güte ist stabil** | Die Gewichte stammen aus dem Fenster der letzten Ursprünge; wechselt das beste Mitglied schneller, hinken sie hinterher. | Zerfall im adaptiven Verfahren |
| **Die Mitglieder machen verschiedene Fehler** | Sind die Fehler stark korreliert, gewinnt die Kombination wenig. Der Pool hier mischt bewusst verschiedene Familien. | Vielfältige Mitglieder |
| **Punktprognosen genügen** | Kombiniert wird hier ein Wert je Tag; Intervalle (Stück 7) und abgestimmte Hierarchien (Stück 8) lassen sich ebenfalls kombinieren, sind aber nicht gebaut. | Quantilkombination |
| **Der Vergleich mit dem besten Mitglied ist fair** | Das beste Mitglied im Nachhinein kennt niemand vorher; der Vorsprung der Kombination gilt gegen einen nicht erreichbaren Maßstab. Ein zufällig gewähltes Mitglied wäre deutlich schlechter. | – |
| **Erzeugtes Portfolio, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, AR(1)-Schwankungen); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, 56 Tests, rund zwei Minuten wegen der Experimente): die Kombinationen von Hand und gegen unabhängige Schleifen und `scipy` (Simplex-Löser samt KKT, Kleinste Quadrate, Fenster, adaptives Verfahren), Eigenschaften (nur realisierte Fehler, Verteilungen, Grenzfälle), das Portfolio, die Mitglieder von Hand, kein Blick in die Zukunft von Mitgliedern und Kombinationen, Kennzahlen und Referenzen von Hand,
Preset- und Permalink-Klemmen, AppTest-Rauchtests (Standard, jedes Preset, Depot- und Ursprungs-Regler, zu kleiner Pool und leere Auswahl, Extremwerte, drei Experimente auf Abruf, keine unaufgelösten Platzhalter) und `test_claims.py` (jede Zahl aus diesem README und aus den Preset-Hinweisen mit Bändern und Rangfolgen).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `cmb_constants.py` | Regler-Grenzen, Mitglieder, Verfahren, Experiment-Seeds |
| `cmb_presets.py` | Permalink/Presets-Mechanik, `PRESET_HELP` |
| `cmb_scenario.py` | Das Portfolio (Depots, Niveauschwankungen) |
| `cmb_members.py` | Die sechs Mitglieder und ihre Prognosen |
| `cmb_baselines.py`, `cmb_ets.py` | Wochenmittel, Holt-Winters, Regression je Depot |
| `cmb_tree.py`, `cmb_gbm.py`, `cmb_features.py` | Das globale Boosting (aus Stück 6) |
| `cmb_combine.py` | Die acht Kombinationsverfahren |
| `cmb_evaluation.py` | Analyse, Kennzahlen, Aufschlüsselungen, drei Experimente |
| `cmb_visualization.py` | Plotly-Abbildungen |

## Bewusst nicht umgesetzt

- ARIMA und weitere Mitglieder; Mitglieder mit eigener Unsicherheit (Intervalle kombinieren).
- Gewichte, die von Horizont, Wochentag oder Depotgruppe abhängen; Schrumpfung der geschätzten Gewichte zum Mittelwert; Stacking mit einem lernenden Modell.
- Kombination abgestimmter hierarchischer Prognosen (Stück 8).
- Ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy (Gegenprobe im Test: scipy).
