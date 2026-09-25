"""Prognosekombination - viele Prognosen zu einer - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Zeitreihen-Prognose-Linie der "Konzepte"-Reihe: der Zusammenfluss der Verfahren der Vorgänger. Sechs Prognoseverfahren bilden einen Pool; acht Arten, ihre Prognosen zu EINER zu machen
(einfacher Mittelwert, Median, getrimmter Mittelwert, Inverse-MSE-Gewichte, Kleinste-Quadrate-Gewichte frei und nicht negativ, adaptive Gewichte, Auswahl des Besten), gemessen gegen die einzelnen Mitglieder.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import cmb_constants as C
from cmb_evaluation import Settings, analyse, pool_experiment, swing_experiment, window_experiment
from cmb_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from cmb_visualization import SHORT, build_best_counts, build_horizon, build_mase, build_pool, build_series, build_swing, build_time, build_weights, build_window

st.set_page_config(page_title="Prognosekombination – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def spct(x, digits=1):
    """Prozent mit ausdrücklichem Vorzeichen (Minuszeichen U+2212)."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return ("+" if x > 0 else "−" if x < 0 else "±") + f"{abs(x):.{digits}f}".replace(".", ",") + " %"


def rel(v, ref):
    return 100.0 * (v / ref - 1.0)


@st.cache_data(show_spinner=False)
def _swing(levels, seeds):
    return swing_experiment(levels, seeds)


@st.cache_data(show_spinner=False)
def _pool(sizes, seeds):
    return pool_experiment(sizes, seeds)


@st.cache_data(show_spinner=False)
def _window(levels, seeds):
    return window_experiment(levels, seeds)


st.title("🧩 Prognosekombination – viele Prognosen zu einer")
st.markdown(
    """
Wer eine Nachfrage prognostiziert, hat meist mehrere Modelle zur Hand: das Wochenmittel, Holt-Winters, die Regression auf Kalender und Aktionen, ein Boosting-Modell über alle Depots. Welches ist das beste? Das weiß man erst **hinterher**. Die Alternative ist, **alle zu verwenden** und ihre Prognosen zu **kombinieren**:
als einfachen Mittelwert, als Median, mit Gewichten nach vergangener Güte oder mit Gewichten, die man aus den letzten Fehlern schätzt. Die Demo nimmt sechs Verfahren der Vorgänger als **Pool** und misst auf einem **Portfolio von Depots**, **wann die Kombination** dem besten Mitglied nahe kommt oder es sogar schlägt,
**wann der einfache Mittelwert genügt** und was ein schlechtes Mitglied im Pool anrichtet. Alle Daten sind erzeugt; die Rechnung ist in numpy geschrieben.
"""
)
st.caption(
    "Neuntes Stück der **Zeitreihen-Prognose-Linie** der \"Konzepte\"-Reihe: der **Zusammenfluss** der Verfahren aus den Stücken 1, 2, 4 und 6. **Bezug zu OR:** Prognosen sind Eingaben von Planungsmodellen; ein robuster Verbund ist oft wertvoller als das zufällig beste Einzelmodell, "
    "zumal man das beste nur nachträglich kennt."
)

with st.expander("So wird kombiniert", expanded=True):
    st.markdown(
        """
1. **Der Pool.** Sechs Mitglieder: Naiv (letzter Wert), Letzte Woche, Wochenmittel (Stück 1), Holt-Winters (Stück 2), Regression auf Kalender und Aktionsplan (Stück 4) und das **globale Boosting** (Stück 6). Alle schätzen ihre Parameter auf den Tagen vor Tag 610; die Tage 610–729 sind das **Kalibrierfenster**, das Testjahr beginnt bei Tag 730.
2. **Feste Gewichte.** **Mittelwert** (alle $1/K$), **Median**, **getrimmter Mittelwert** (ohne das kleinste und größte Mitglied, ab vier Mitgliedern): keine Schätzung, also kein Schätzfehler.
3. **Geschätzte Gewichte.** Zu einem Testursprung $t$ zählen nur Prognosen, deren Zieltag schon bekannt ist, aus den letzten $W$ Ursprüngen: **Inverse MSE** ($w_k \\propto 1/\\text{MSE}_k$, Bates/Granger), **Kleinste Quadrate** (frei, Granger/Ramanathan: $\\min_w \\sum (y - w'F)^2$),
   **Kleinste Quadrate nicht negativ mit Summe 1** (auf dem Simplex), **adaptiv** ($w_k \\propto D_k^{-\\eta}$ mit dem exponentiell abklingenden Fehler $D_k$; $\\eta = 0$ ist der Mittelwert, $\\eta = 1$ Inverse MSE, groß ist die Auswahl) und **Auswahl** (das Mitglied mit dem kleinsten Fehler im Fenster).
4. **Relative Fehler.** Ist und Prognosen werden durch die mittlere Mitglieder-Prognose geteilt, damit starke Wochentage die Gewichte nicht dominieren.
5. **Kennzahl.** MASE je Depot (MAE geteilt durch den saisonal naiven Trainingsfehler), über die Depots gemittelt; Referenzen: das beste Mitglied **im Nachhinein** und das beste Mitglied **je Depot im Nachhinein** (beide erst nach dem Testjahr bekannt).
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Portfolio**")
    n_depots = st.slider("Zahl der Depots", *bounds("depots_slider"), key="depots_slider", step=C.DEPOTS_STEP, help="Wie viele Depots das Portfolio hat; das Boosting lernt aus allen.")
    noise = st.slider("Rauschen (Mittel der Depots)", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Mittlere Streuung des multiplikativen Tagesrauschens.")
    trend = st.slider("Trend (% je Jahr, Mittel)", *bounds("trend_slider"), key="trend_slider", step=C.TREND_STEP, help="Mittleres Wachstum der Depots in Prozent des Ausgangsniveaus je Jahr.")
    swing = st.slider("Niveauschwankung der Depots", *bounds("swing_slider"), key="swing_slider", step=C.SWING_STEP, help="Stärke der langsamen, depoteigenen Niveauschwankungen (im Log; je Depot mit eigenem Faktor). Die statische Regression kann ihnen nicht folgen, die anpassungsfähigen Verfahren schon.")
    st.markdown("**Der Pool**")
    members = st.multiselect("Mitglieder", list(C.MEMBERS), key="members_select", format_func=lambda m: C.MEMBER_NAMES[m], help="Welche Verfahren im Pool sind (mindestens zwei; bei weniger nimmt die App die Voreinstellung).")
    horizon = st.slider("Prognosehorizont (Tage)", *bounds("horizon_slider"), key="horizon_slider", help="Wie viele Tage im Voraus prognostiziert wird.")
    st.markdown("**Die Gewichte**")
    window = st.slider("Kalibrierfenster (Ursprünge)", *bounds("window_slider"), key="window_slider", step=C.WINDOW_STEP, help="Aus den letzten so vielen realisierten Ursprüngen werden die Gewichte geschätzt (auch der Zerfall des adaptiven Verfahrens: 1 − 1/Fenster).")
    eta = st.slider("Schärfe η des adaptiven Verfahrens", *bounds("eta_slider"), key="eta_slider", step=C.ETA_STEP, help="Gewicht ∝ Fehler^(−η): 0 = Mittelwert, 1 = Inverse MSE, groß = Auswahl des Besten.")
    st.markdown("**Anzeige**")
    methods = st.multiselect("Kombinationen in den Diagrammen", list(C.METHODS), key="methods_select", format_func=lambda m: C.METHOD_NAMES[m], help="Welche Kombinationen im Depot-, Horizont- und Zeitdiagramm dick gezeigt werden.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt das ganze Portfolio fest.")
    st.button("🎲 Neues Portfolio generieren", width="stretch", on_click=randomize_seed)

pool = [m for m in C.MEMBERS if m in members]
if len(pool) < 2:
    pool = list(C.DEFAULT_MEMBERS)
shown = [m for m in C.METHODS if m in methods] or ["mean", "simplex"]
sync_query_params({"depots_slider": int(n_depots), "noise_slider": round(float(noise), 2), "trend_slider": int(trend), "swing_slider": round(float(swing), 2), "horizon_slider": int(horizon), "members_select": pool,
                   "window_slider": int(window), "eta_slider": round(float(eta), 1), "methods_select": shown, "seed_input": int(seed)})

settings = Settings(int(n_depots), round(float(noise), 2), int(trend), round(float(swing), 2), int(horizon), tuple(pool), int(window), round(float(eta), 1), int(seed))
if len(members) < 2:
    st.warning("Weniger als zwei Mitglieder gewählt: die App nimmt den Pool der Voreinstellung (Wochenmittel, Holt-Winters, Regression, Boosting).")
if not methods:
    st.warning("Keine Kombination gewählt: die Diagramme zeigen Mittelwert und Kleinste Quadrate (nicht negativ).")
with st.spinner("Die Mitglieder werden prognostiziert und kombiniert ..."):
    a = analyse(settings)
port, sm = a.port, a.summary
names = list(a.members)

# --- Ein Depot -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ein Depot und die Prognosen an einem Ursprung")
st.session_state["depot_slider"] = min(port.n - 1, max(0, st.session_state.get("depot_slider", 0)))
dep = int(st.slider("Depot", 0, port.n - 1, key="depot_slider", help="Welches Depot des Portfolios gezeigt wird."))
lo_o, hi_o = int(a.test_org[0]), int(a.test_org[-1])
st.session_state["origin_slider"] = min(hi_o, max(lo_o, st.session_state.get("origin_slider", 900)))
origin = int(st.slider("Ursprung (Tag)", lo_o, hi_o, key="origin_slider", help="Ab diesem Tag wird prognostiziert; bekannt ist alles davor. Alle Ursprünge des Testjahres gehen in die Auswertung ein."))
st.plotly_chart(build_series(a, dep, origin, shown), width="stretch", key="series_chart")
st.caption(
    f"Depot {dep}: Niveau {de(port.level[dep], 0)} Aufträge, Schwankung {de(port.swing[dep], 2)}. Gepunktet die {len(names)} Mitglieder, dick die gewählten Kombinationen. MASE in diesem Depot: "
    + ", ".join(f"{SHORT[m]} {de(a.mase[m][dep], 2)}" for m in names) + "; "
    + ", ".join(f"{SHORT[m]} {de(a.mase[m][dep], 2)}" for m in shown) + "."
)
wm_options = [m for m in C.METHODS if m in a.weights and a.weights[m] is not None]
wsel = st.selectbox("Gewichte über die Zeit für", wm_options, index=wm_options.index(next((m for m in shown if m in wm_options), wm_options[0])), format_func=lambda m: C.METHOD_NAMES[m], key="weights_select",
                    help="Welches Verfahren mit seinen Gewichten je Ursprung für das gewählte Depot gezeigt wird (Median und getrimmter Mittelwert haben keine festen Gewichte).")
st.plotly_chart(build_weights(a, dep, wsel), width="stretch", key="weights_chart")
st.caption("Die Gewichte, mit denen die Mitglieder-Prognosen des Depots an jedem Testursprung zur Kombination werden (gestapelt bis 1; bei freien Kleinsten Quadraten einzeln, auch negativ). Sie stammen nur aus Fehlern, die zu diesem Ursprung schon bekannt waren.")

st.markdown("---")

# --- Auswertung -----------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was bringt die Kombination?")
best_m = a.best_member
ref = sm[best_m]
rows = []
for k in names + list(a.forecasts):
    rows.append({"Verfahren": ("Mitglied: " + C.MEMBER_NAMES[k]) if k in names else "Kombination: " + C.METHOD_NAMES[k], "MASE": de(sm[k], 3), "gegenüber dem besten Mitglied": spct(rel(sm[k], ref), 1) if k != best_m else "–"})
rows.append({"Verfahren": "Bestes Mitglied je Depot im Nachhinein", "MASE": de(a.oracle_depot, 3), "gegenüber dem besten Mitglied": spct(rel(a.oracle_depot, ref), 1)})
st.dataframe(rows, hide_index=True)
combos = list(a.forecasts)
best_c = min(combos, key=lambda m: sm[m])
if sm[best_c] < ref - 1e-12:
    st.success(f"✅ Die beste Kombination ({SHORT[best_c]}, {de(sm[best_c], 3)}) schlägt das beste Mitglied im Nachhinein ({SHORT[best_m]}, {de(ref, 3)}) um {de(-rel(sm[best_c], ref), 1)} %; der einfache Mittelwert liegt bei {de(sm['mean'], 3)} ({spct(rel(sm['mean'], ref), 1)}), "
               f"die Auswahl des Besten bei {de(sm['select'], 3)} ({spct(rel(sm['select'], ref), 1)}).")
else:
    st.info(f"Keine Kombination schlägt das beste Mitglied im Nachhinein ({SHORT[best_m]}, {de(ref, 3)}); die beste ({SHORT[best_c]}) liegt bei {de(sm[best_c], 3)} ({spct(rel(sm[best_c], ref), 1)}), der Mittelwert bei {de(sm['mean'], 3)} ({spct(rel(sm['mean'], ref), 1)}). "
            "Wer das beste Mitglied vorher nicht kennt, zahlt für die Sicherheit dieses Aufschlags.")
st.caption(
    f"MASE über {len(a.test_org)} Ursprünge, {settings.horizon} Horizonte und {port.n} Depots. Das beste Mitglied im Nachhinein ({SHORT[best_m]}) steht erst nach dem Testjahr fest; das beste Mitglied je Depot im Nachhinein ist noch schärfer (es wählt für jedes Depot getrennt) und für keine Methode erreichbar. "
    f"Kalibrierfenster {settings.window} Ursprünge, η = {de(settings.eta, 0)}."
)
st.plotly_chart(build_mase(a), width="stretch", key="mase_chart")
st.markdown("##### Wo ist welches Mitglied am besten?")
st.plotly_chart(build_best_counts(a), width="stretch", key="counts_chart")
st.markdown("##### Wie der Fehler mit dem Horizont wächst")
st.plotly_chart(build_horizon(a, shown), width="stretch", key="horizon_chart")
st.markdown("##### Im Zeitverlauf des Testjahres")
st.plotly_chart(build_time(a, shown), width="stretch", key="time_chart")
counts = {m: int(np.sum(np.argmin(np.stack([a.mase[k] for k in names]), axis=0) == i)) for i, m in enumerate(names)}
st.caption(f"Das beste Mitglied wechselt von Depot zu Depot: " + ", ".join(f"{SHORT[m]} in {counts[m]}" for m in names) + f" von {port.n} Depots. Gerade dann ist die Kombination interessant: sie muss das beste Mitglied nicht vorher kennen.")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wann ist der Mittelwert genug? Wachsende Niveauschwankung")
st.caption(f"{C.EXP_DEPOTS} Depots, Pool der Voreinstellung; die depoteigene Niveauschwankung wächst auf {', '.join(str(x).replace('.', ',') for x in C.SWING_LEVELS)}. Gezeigt: MASE der Mitglieder (gepunktet) und einiger Kombinationen. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds (Fehlerbalken: Standardfehler). Dauer etwa eine Minute.")
if st.button("Niveauschwankung durchrechnen", key="swing_start"):
    st.session_state["swing_on"] = True
if st.session_state.get("swing_on"):
    rs = _swing(C.SWING_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_swing(rs, C.DEFAULT_MEMBERS), width="stretch", key="swing_chart")
    r0, r1 = rs[0]["mase"], rs[-1]["mase"]
    best0 = min(C.DEFAULT_MEMBERS, key=lambda m: r0[m][0])
    best1 = min(C.DEFAULT_MEMBERS, key=lambda m: r1[m][0])
    st.warning(
        f"**Befund:** Ohne Schwankung ist die Regression das beste Mitglied ({de(r0[best0][0], 3)}, denn sie kennt die wahre Form); der einfache Mittelwert liegt mit {de(r0['mean'][0], 3)} darüber ({spct(rel(r0['mean'][0], r0[best0][0]), 1)}), die Gewichte nach Kleinsten Quadraten "
        f"({de(r0['simplex'][0], 3)}) und die adaptiven ({de(r0['ewa'][0], 3)}) erreichen es ungefähr. Bei Schwankung {de(rs[-1]['swing'], 2)} fällt die statische Regression auf {de(r1['reg'][0], 3)}; bestes Mitglied ist jetzt {SHORT[best1]} ({de(r1[best1][0], 3)}), der einfache Mittelwert {de(r1['mean'][0], 3)} "
        f"({spct(rel(r1['mean'][0], r1[best1][0]), 1)}), der Median {de(r1['median'][0], 3)}, adaptiv {de(r1['ewa'][0], 3)} ({spct(rel(r1['ewa'][0], r1[best1][0]), 1)}). **Das beste Mitglied wechselt mit den Umständen - die Kombination folgt ihm, ohne es zu wissen; die Auswahl nach dem Fenster ({de(r1['select'][0], 3)}) folgt ihm schlechter.**"
    )

st.markdown("---")

st.subheader("🔬 Ein schlechtes Mitglied im Pool")
st.caption(f"{C.EXP_DEPOTS} Depots, Schwankung 0,06; der Pool wächst von zwei bis {C.POOL_SIZES[-1]} Mitglieder, vom besten zum schlechtesten hinzugefügt (Regression, Boosting, Holt-Winters, Wochenmittel, Letzte Woche, Naiv). Gezeigt: MASE der Kombinationen und das beste Mitglied des Pools. "
           f"Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Pool durchrechnen", key="pool_start"):
    st.session_state["pool_on"] = True
if st.session_state.get("pool_on"):
    rp = _pool(C.POOL_SIZES, C.EXP_SEEDS)
    st.plotly_chart(build_pool(rp), width="stretch", key="pool_chart")
    p0, p1 = rp[0], rp[-1]
    st.warning(
        f"**Befund:** Mit den zwei besten Mitgliedern liegt der Mittelwert bei {de(p0['methods']['mean'][0], 3)}, das beste Mitglied bei {de(p0['best'][0], 3)}. Mit allen {p1['k']} Mitgliedern steigt der Mittelwert auf {de(p1['methods']['mean'][0], 3)} - der schlechteste Wert des Pools (Naiv) zieht ihn nach oben; "
        f"der Median bleibt bei {de(p1['methods']['median'][0], 3)}, die Gewichte nach Kleinsten Quadrate (nicht negativ) bei {de(p1['methods']['simplex'][0], 3)} und die adaptiven bei {de(p1['methods']['ewa'][0], 3)}: **wer die Güte schätzt oder robust mittelt, verträgt schlechte Mitglieder; der einfache Mittelwert nicht.**"
    )

st.markdown("---")

st.subheader("🔬 Wie viele Daten brauchen die Gewichte?")
st.caption(f"{C.EXP_DEPOTS} Depots, Pool der Voreinstellung; Kalibrierfenster von {', '.join(str(w) for w in C.WINDOW_LEVELS)} Ursprüngen. Der Mittelwert schätzt nichts und ist vom Fenster unabhängig. Mittel über {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Kalibrierfenster durchrechnen", key="window_start"):
    st.session_state["window_on"] = True
if st.session_state.get("window_on"):
    rw = _window(C.WINDOW_LEVELS, C.EXP_SEEDS)
    st.plotly_chart(build_window(rw), width="stretch", key="window_chart")
    w0, w1 = rw[0]["mase"], rw[-1]["mase"]
    st.warning(
        f"**Befund:** Mit {rw[0]['window']} Ursprüngen im Fenster sind die freien Gewichte verrauscht: MASE {de(w0['ols'][0], 3)} gegen {de(w1['ols'][0], 3)} bei {rw[-1]['window']}, während die Beschränkung auf nicht negative Gewichte mit Summe 1 stabil bleibt ({de(w0['simplex'][0], 3)} gegen {de(w1['simplex'][0], 3)}). "
        f"Der Mittelwert ({de(w0['mean'][0], 3)}) hängt nicht vom Fenster ab. Das ist das **Kombinations-Rätsel**: Gewichte zu schätzen kostet Schätzfehler, der einfache Mittelwert spart ihn - und ist deshalb eine starke Vergleichsgröße."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Mitglieder sind ähnlich gut** | Ist eines viel schlechter, zieht es den einfachen Mittelwert nach unten (Experiment); Median, getrimmter Mittelwert und geschätzte Gewichte sind robuster. | Pool vorher prüfen, Gewichte schätzen |
| **Gewichte lassen sich schätzen** | Aus kurzen Fenstern sind sie verrauscht, besonders frei geschätzte; die Schätzung kann schlechter sein als der einfache Mittelwert (Kombinations-Rätsel). | Beschränkung (nicht negativ, Summe 1), größeres Fenster, Schrumpfung zum Mittelwert |
| **Die Güte ist stabil** | Die Gewichte stammen aus dem Fenster der letzten Ursprünge; wechselt das beste Mitglied schneller, hinken sie hinterher. | Zerfall im adaptiven Verfahren |
| **Die Mitglieder machen verschiedene Fehler** | Sind die Fehler stark korreliert (gleiche Datenbasis, ähnliche Modelle), gewinnt die Kombination wenig. | Vielfältige Mitglieder |
| **Punktprognosen genügen** | Kombiniert wird hier ein Wert je Tag; Prognoseintervalle (Stück 7) und abgestimmte Hierarchien (Stück 8) lassen sich ebenfalls kombinieren, sind aber nicht gebaut. | Quantilkombination, Kombination abgestimmter Prognosen |
| **Erzeugtes Portfolio, drei Seeds** | Das Vehikel erzeugt genau die Muster (multiplikativ, log-normal, AR(1)-Schwankungen); echte Portfolios sind unordentlicher. Die Zahlen gelten für diese Portfolios. | – |
"""
)
st.caption("Die Linie: Naive Prognose → Exponentielle Glättung → ARIMA → Dynamische Regression, dazu Croston, Boosting, Prognoseintervalle, Hierarchie, **Kombination**, Bestand und ein vortrainiertes Netz (die übrigen Stücke noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kombination.** $K$ Mitglieder-Prognosen $F_k$ für dasselbe Depot, denselben Ursprung und Horizont; $\hat y = \sum_k w_k F_k$. Mittelwert: $w_k = 1/K$. Median und getrimmter Mittelwert: Ordnungsstatistiken von $F_1, \dots, F_K$ (getrimmt: ohne die $\lfloor K/4 \rfloor$ kleinsten und größten).

**Relative Fehler.** $s = \bar F + 1$ mit dem Mittel der Mitglieder; alle Schätzungen mit $\tilde F_k = F_k / s$, $\tilde y = y / s$. Zum Testursprung $t$ und Horizont $h$ zählen die Ursprünge $o \in (t - h - W,\ t - h]$ (Zieltage $o + j \le t - 1$), mit den Summen über ihre Horizonte:
$A = \sum \tilde F \tilde F'$, $b = \sum \tilde F \tilde y$, $\text{MSE}_k = \frac1{m}\sum (\tilde y - \tilde F_k)^2$ ($m$ Zeilen im Fenster).

**Inverse MSE:** $w_k = \text{MSE}_k^{-1} / \sum_l \text{MSE}_l^{-1}$. **Auswahl:** $w = e_{\arg\min \text{MSE}_k}$. **Kleinste Quadrate frei:** $w = (A/m + \rho I)^{-1} b/m$ mit winziger Ridge-Strafe $\rho$. **Nicht negativ, Summe 1:** $\min_w\; w'(A/m)w - 2 (b/m)'w$ unter $w \ge 0$, $\mathbf 1' w = 1$, gelöst **exakt**: alle Träger (Teilmengen der Mitglieder) werden durchprobiert, je Träger gilt das KKT-System, zulässig sind die nicht negativen Lösungen, es zählt der kleinste Zielwert.
**Adaptiv:** $D_{k,o} = \lambda D_{k,o-1} + \overline{\text{MSE}}_{k,o}$ mit $\lambda = 1 - 1/W$, $w_k \propto D_k^{-\eta}$.

**Kennzahl.** MASE je Depot: MAE über alle Ursprünge des Testjahres und Horizonte, geteilt durch den mittleren absoluten saisonal naiven Fehler (Periode 7) auf den Tagen vor 730; über die Depots gemittelt. "Bestes Mitglied je Depot im Nachhinein": Mittel über die Depots der kleinsten Mitglieder-MASE.

Implementiert in `cmb_members.py` (die sechs Mitglieder), `cmb_combine.py` (die Kombinationen), `cmb_baselines.py`, `cmb_ets.py`, `cmb_tree.py`, `cmb_gbm.py`, `cmb_features.py` (Verfahren aus den Vorgängern), `cmb_scenario.py` (das Portfolio), `cmb_evaluation.py` (Analyse, drei Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
