"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Depot-/Ursprungs-Regler, Würfel-Knopf, Permalink-Grenzen, Extremwerte, drei Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import cmb_constants as C
import cmb_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=600)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value, el.value[:120]


def test_default_run_shows_charts_and_a_verdict():
    at = _run()
    _ok(at)
    assert len(at.get("plotly_chart")) == 6 and len(at.info) + len(at.success) + len(at.warning) >= 1


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]


def test_depot_and_origin_selections_survive_a_smaller_portfolio_and_longer_horizon():
    at = _run(depot_slider=29, origin_slider=1094 - 14)
    _ok(at)
    at.slider(key="depots_slider").set_value(10).run()
    _ok(at)
    assert at.session_state["depot_slider"] <= 9
    at.slider(key="horizon_slider").set_value(28).run()
    _ok(at)
    assert at.session_state["origin_slider"] <= C.N_DAYS - 28


def test_too_small_pool_and_empty_method_selection_fall_back():
    at = _run(members_select=["gbm"], methods_select=[])
    _ok(at)
    assert any("Weniger als zwei Mitglieder" in w.value for w in at.warning) and any("Keine Kombination gewählt" in w.value for w in at.warning)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Portfolio generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=600)
    at.query_params["depots"] = "77"
    at.query_params["window"] = "10"
    at.query_params["eta"] = "99"
    at.query_params["members"] = "gbm,reg"
    at.query_params["show"] = "quatsch"
    at.query_params["swing"] = "abc"
    at.run()
    _ok(at)
    assert at.session_state["depots_slider"] == 80 and at.session_state["window_slider"] == C.WINDOW_MIN and at.session_state["eta_slider"] == C.ETA_MAX
    assert at.session_state["members_select"] == ["reg", "gbm"] and at.session_state["methods_select"] == ["mean", "invmse", "simplex", "ewa"] and at.session_state["swing_slider"] == C.DEFAULT_SWING


@pytest.mark.parametrize("kw", [dict(depots_slider=C.DEPOTS_MIN, horizon_slider=1, window_slider=C.WINDOW_MIN), dict(horizon_slider=C.HORIZON_MAX, window_slider=C.WINDOW_MAX, eta_slider=C.ETA_MAX),
                                dict(swing_slider=C.SWING_MAX, noise_slider=C.NOISE_MAX, trend_slider=C.TREND_MIN), dict(members_select=list(C.MEMBERS), methods_select=list(C.METHODS)),
                                dict(members_select=["naive", "snaive1"], eta_slider=0.0, swing_slider=0.0)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def _small(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0,))
    monkeypatch.setattr(C, "EXP_DEPOTS", 10)
    monkeypatch.setattr(C, "SWING_LEVELS", (0.0, 0.12))
    monkeypatch.setattr(C, "POOL_SIZES", (2, 6))
    monkeypatch.setattr(C, "WINDOW_LEVELS", (30, 120))


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()
    _ok(at)


def test_swing_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "swing_start")
    assert at.session_state["swing_on"] and any("Das beste Mitglied wechselt mit den Umständen" in w.value for w in at.warning)


def test_pool_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "pool_start")
    assert at.session_state["pool_on"] and any("verträgt schlechte Mitglieder" in w.value for w in at.warning)


def test_window_experiment_runs_on_demand(monkeypatch):
    _small(monkeypatch)
    at = _run()
    _click(at, "window_start")
    assert at.session_state["window_on"] and any("Kombinations-Rätsel" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
