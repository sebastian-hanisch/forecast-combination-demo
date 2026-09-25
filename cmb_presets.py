"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. hrc_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import cmb_constants as C


def _choice_list(allowed):
    def caster(value):
        parts = [str(v) for v in value] if isinstance(value, (list, tuple)) else [v.strip() for v in str(value).split(",") if v.strip()]
        if not parts or any(p not in allowed for p in parts):
            raise ValueError(value)
        return [m for m in allowed if m in parts]
    return caster


_members = _choice_list(C.MEMBERS)
_methods = _choice_list(C.METHODS)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "depots_slider": SettingSpec("depots", int, C.DEFAULT_DEPOTS, C.DEPOTS_MIN, C.DEPOTS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "trend_slider": SettingSpec("trend", int, C.DEFAULT_TREND, C.TREND_MIN, C.TREND_MAX),
    "swing_slider": SettingSpec("swing", float, C.DEFAULT_SWING, C.SWING_MIN, C.SWING_MAX),
    "horizon_slider": SettingSpec("horizon", int, C.DEFAULT_HORIZON, C.HORIZON_MIN, C.HORIZON_MAX),
    "members_select": SettingSpec("members", _members, list(C.DEFAULT_MEMBERS)),
    "window_slider": SettingSpec("window", int, C.DEFAULT_WINDOW, C.WINDOW_MIN, C.WINDOW_MAX),
    "eta_slider": SettingSpec("eta", float, C.DEFAULT_ETA, C.ETA_MIN, C.ETA_MAX),
    "methods_select": SettingSpec("show", _methods, ["mean", "invmse", "simplex", "ewa"]),
    "seed_input": SettingSpec("seed", int, 3, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n_depots": "depots_slider", "noise": "noise_slider", "trend": "trend_slider", "swing": "swing_slider", "horizon": "horizon_slider", "members": "members_select", "window": "window_slider", "eta": "eta_slider",
               "show": "methods_select", "seed": "seed_input"}
STEPS = {"depots_slider": C.DEPOTS_STEP, "noise_slider": C.NOISE_STEP, "trend_slider": C.TREND_STEP, "swing_slider": C.SWING_STEP, "window_slider": C.WINDOW_STEP, "eta_slider": C.ETA_STEP}


def _p(**kw):
    base = {"n_depots": C.DEFAULT_DEPOTS, "noise": C.DEFAULT_NOISE, "trend": C.DEFAULT_TREND, "swing": C.DEFAULT_SWING, "horizon": C.DEFAULT_HORIZON, "members": list(C.DEFAULT_MEMBERS), "window": C.DEFAULT_WINDOW,
            "eta": C.DEFAULT_ETA, "show": ["mean", "invmse", "simplex", "ewa"], "seed": 3}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall: vier Mitglieder, 30 Depots": _p(),
    "Ohne Schwankung: die Regression ist die Wahrheit": _p(swing=0.0, show=["mean", "invmse", "simplex", "select"]),
    "Starke Schwankung (0,12)": _p(swing=0.12, show=["mean", "median", "simplex", "ewa"]),
    "Ein schlechtes Mitglied im Pool (Naiv)": _p(members=["naive", "wm", "hw", "reg", "gbm"], show=["mean", "median", "trim", "ewa"]),
    "Kleines Kalibrierfenster (30)": _p(window=30, show=["mean", "ols", "simplex", "ewa"]),
    "Nur Holt-Winters und Regression": _p(members=["hw", "reg"], show=["mean", "invmse", "simplex", "select"]),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = list(spec.default) if isinstance(spec.default, list) else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 3)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = ",".join(value) if isinstance(value, (list, tuple)) else str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        v = PRESETS[name][key]
        st.session_state[state_key] = list(v) if isinstance(v, list) else v


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall: vier Mitglieder, 30 Depots": "Pool aus Wochenmittel, Holt-Winters, Regression und Boosting (Seed 3, Schwankung 0,06): bestes Mitglied im Nachhinein Boosting 0,833 (in 22 von 30 Depots das beste, die Regression in 8); Mittelwert 0,842 (+1,0 %), Inverse MSE −0,8 %, Kleinste Quadrate (nicht negativ) −1,1 %, adaptiv 0,815 (−2,2 %), Auswahl +2,5 %.",
    "Ohne Schwankung: die Regression ist die Wahrheit": "Ohne Schwankung ist die Regression das beste Mitglied (0,775): Kleinste Quadrate 0,744 (−4,0 %) und adaptiv 0,745 (−3,8 %) schlagen sie, der Mittelwert 0,780 (+0,7 %) nicht; die Auswahl 0,758 (−2,1 %).",
    "Starke Schwankung (0,12)": "Schwankung 0,12: die Regression fällt auf 1,712, bestes Mitglied ist das Boosting (0,906); Mittelwert 0,953 (+5,2 %), Median 0,908 (+0,2 %), adaptiv 0,894 (−1,4 %), Kleinste Quadrate frei 0,950 (+4,9 %).",
    "Ein schlechtes Mitglied im Pool (Naiv)": "Mit dem Mitglied Naiv (MASE 2,853) im Pool steigt der Mittelwert auf 1,045 (+25,5 % gegen das beste Mitglied, Boosting 0,833); Median 0,853 (+2,4 %), getrimmt 0,850 (+2,1 %), adaptiv 0,814 (−2,3 %), Kleinste Quadrate (nicht negativ) 0,825 (−1,0 %).",
    "Kleines Kalibrierfenster (30)": "Fenster von 30 Ursprüngen: Kleinste Quadrate frei 0,865 (+3,9 % gegen das beste Mitglied 0,833), nicht negativ 0,827 (−0,7 %), adaptiv 0,817 (−1,9 %); der Mittelwert 0,842 hängt nicht vom Fenster ab.",
    "Nur Holt-Winters und Regression": "Zwei Mitglieder (Holt-Winters 0,887, Regression 1,148): der Mittelwert 0,897 (+1,2 %), Kleinste Quadrate frei 0,834 (−5,9 %), adaptiv 0,850 (−4,1 %), Inverse MSE 0,852 (−3,9 %); die Mitglieder ergänzen sich (Holt-Winters ist in 18 Depots besser, die Regression in 12).",
}
