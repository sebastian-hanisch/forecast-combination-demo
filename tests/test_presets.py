"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen und Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import cmb_constants as C
import cmb_evaluation as E
import cmb_presets as P


def _settings(p):
    return E.Settings(p["n_depots"], p["noise"], p["trend"], p["swing"], p["horizon"], tuple(p["members"]), p["window"], p["eta"], p["seed"])


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi
        for key, state_key in (("n_depots", "depots_slider"), ("noise", "noise_slider"), ("trend", "trend_slider"), ("swing", "swing_slider"), ("window", "window_slider"), ("eta", "eta_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-6
        assert len(p["members"]) >= 2 and set(p["members"]) <= set(C.MEMBERS) and p["show"] and set(p["show"]) <= set(C.METHODS)


def test_standard_preset_equals_the_default_settings():
    assert _settings(P.PRESETS["Standardfall: vier Mitglieder, 30 Depots"]) == E.Settings()


def test_bounds_steps_and_unique_url_params():
    assert P.bounds("window_slider") == (C.WINDOW_MIN, C.WINDOW_MAX) and P.bounds("eta_slider") == (C.ETA_MIN, C.ETA_MAX)
    assert set(P.STEPS) == {"depots_slider", "noise_slider", "trend_slider", "swing_slider", "window_slider", "eta_slider"}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_casters_reject_bad_values_and_canonicalise_lists():
    for caster, bad in ((P._members, ""), (P._members, "hw,quatsch"), (P._methods, "median,mittel")):
        try:
            caster(bad)
        except ValueError:
            continue
        raise AssertionError(bad)
    assert P._members("gbm,reg") == ["reg", "gbm"] and P._methods(["ewa", "mean"]) == ["mean", "ewa"]
