"""Portfolio, Mitglieder des Pools, Auswertung und ihre Kennzahlen; kein Mitglied und keine Gewichtung darf etwas aus der Zukunft des Ursprungs verwenden."""

import dataclasses

import numpy as np
import pytest

import cmb_baselines as B
import cmb_constants as C
import cmb_evaluation as E
import cmb_members as M
import cmb_scenario as S


def test_scenario_is_deterministic_seeded_and_swing_is_heterogeneous():
    a, b, c = S.generate(6, seed=1, swing=0.06), S.generate(6, seed=1, swing=0.06), S.generate(6, seed=2, swing=0.06)
    assert np.array_equal(a.y, b.y) and not np.array_equal(a.y, c.y) and a.y.shape == (6, C.N_DAYS)
    assert a.swing.min() > 0 and a.swing.max() / a.swing.min() > 1.2 and (S.generate(3, seed=1, swing=0.0).swing == 0).all()


def test_swing_moves_the_level_by_about_its_size():
    flat, wavy = S.generate(20, seed=4, swing=0.0, noise_mean=0.04), S.generate(20, seed=4, swing=0.12, noise_mean=0.04)
    d = np.log((wavy.mu + 1) / (flat.mu + 1))
    assert np.array_equal(flat.promo, wavy.promo) and 0.06 < d.std() < 0.25 and abs(d.mean()) < 0.1


def test_members_by_hand():
    port = S.generate(3, seed=5)
    h = 7
    fc = M.member_forecasts(port, h, members=("naive", "snaive1", "wm"))
    org = M.origins(h)
    y = port.y[1]
    t, j = 700, 4
    i = t - org[0]
    assert fc["naive"][1, i, j] == y[t - 1]
    assert fc["snaive1"][1, i, j] == y[t - 7 + j % 7]
    assert fc["wm"][1, i, j] == pytest.approx(np.mean([y[t - 7 * (k + 1) + j % 7] for k in range(4)]))
    assert fc["wm"].shape == (3, len(org), h) and org[0] == C.FIT_END and org[-1] == C.N_DAYS - h


def test_hw_regression_and_boosting_use_only_the_training_days_for_parameters():
    port = S.generate(3, seed=6)
    h = 3
    fc = M.member_forecasts(port, h)
    y2 = port.y.copy()
    t = 800
    y2[:, t:] = np.random.default_rng(0).integers(0, 999, size=y2[:, t:].shape)
    fc2 = M.member_forecasts(dataclasses.replace(port, y=y2), h)
    i = t - M.origins(h)[0]
    for m in C.MEMBERS:
        assert np.allclose(fc[m][:, :i + 1], fc2[m][:, :i + 1]), m                            # Prognosen bis zum Ursprung t ändern sich nicht
    assert not np.allclose(fc["naive"][:, i + 5:], fc2["naive"][:, i + 5:])                       # die Änderung ist wirksam: spätere Prognosen ändern sich


def test_regression_parameters_do_not_move_when_data_after_fit_end_changes():
    port = S.generate(2, seed=7)
    y = port.y[0]
    X = B.regression_design(port.dow, port.holiday, port.after, port.promo[0])
    org = np.array([650, 720])
    a = B.regression_forecast(y, X, org, 5, first=C.FIT_END)
    y2 = y.copy()
    y2[C.FIT_END:] = 1.0
    assert np.allclose(B.regression_forecast(y2, X, org, 5, first=C.FIT_END), a)


def _settings(**kw):
    return E.Settings(n_depots=10, horizon=7, **kw)


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(_settings())


def test_analysis_shapes_and_definitions(analysis):
    a = analysis
    T = C.N_DAYS - 7 - C.FIRST_TEST + 1
    assert a.actual.shape == (10, T, 7) and set(a.members) == set(C.DEFAULT_MEMBERS) and set(a.forecasts) == set(C.METHODS) and all(f.shape == a.actual.shape for f in a.forecasts.values())
    assert np.allclose(a.forecasts["mean"], np.mean([a.members[m] for m in a.members], axis=0)) and np.allclose(a.forecasts["median"], np.median([a.members[m] for m in a.members], axis=0))
    for m, w in a.weights.items():
        if w is not None:
            assert w.shape == (10, T, 4), m
    assert np.array_equal(a.actual[3, 5], a.port.y[3, a.test_org[5]:a.test_org[5] + 7])


def test_mase_by_hand_and_references(analysis):
    a = analysis
    j = 4
    y = a.port.y[j]
    scale = np.mean(np.abs(y[7:C.FIRST_TEST] - y[:C.FIRST_TEST - 7]))
    assert a.scale[j] == pytest.approx(scale)
    assert a.mase["hw"][j] == pytest.approx(np.mean(np.abs(a.members["hw"][j] - a.actual[j])) / scale)
    stack = np.stack([a.mase[m] for m in a.members])
    assert a.oracle_depot == pytest.approx(stack.min(axis=0).mean()) and a.oracle_depot <= a.summary[a.best_member] + 1e-12 and a.best_member == min(a.members, key=lambda m: a.summary[m])


def test_weights_are_distributions_for_the_constrained_methods(analysis):
    a = analysis
    for m in ("mean", "invmse", "simplex", "ewa", "select"):
        assert (a.weights[m] >= -1e-12).all() and np.allclose(a.weights[m].sum(axis=-1), 1.0), m


def test_breakdowns_average_back(analysis):
    a = analysis
    hz = E.mase_by_horizon(a)
    assert hz["mean"].shape == (7,) and hz["mean"].mean() == pytest.approx(a.summary["mean"])
    td = E.mase_by_target_day(a)
    assert td["mean"].shape == (C.N_DAYS - 7 + 1 - (C.FIRST_TEST + 7 - 1),) and 0.3 < td["mean"].mean() < 3


def test_combination_uses_only_realised_errors_end_to_end():
    s = _settings()
    port = E._portfolio(s.portfolio_key)
    fc = {m: E._members(s.portfolio_key, s.horizon)[m] for m in s.members}
    a1 = E.assemble(port, s, fc)
    t = 850
    Y = port.y.copy()
    Y[:, t:] = 0
    port2 = dataclasses.replace(port, y=Y)
    fc2 = M.member_forecasts(port2, s.horizon, members=s.members)
    a2 = E.assemble(port2, s, fc2)
    i = t - C.FIRST_TEST
    for m in C.METHODS:
        assert np.allclose(a1.forecasts[m][:, :i + 1], a2.forecasts[m][:, :i + 1]), m


def test_horizon_one_and_pool_of_two():
    a = E.analyse(E.Settings(n_depots=10, horizon=1, members=("hw", "reg")))
    assert a.actual.shape[2] == 1 and set(a.members) == {"hw", "reg"} and np.isfinite(a.summary["simplex"]) and np.allclose(a.forecasts["trim"], a.forecasts["mean"])


def test_members_are_computed_once_per_portfolio_and_horizon():
    s = E.Settings(n_depots=10, horizon=3, members=("gbm", "hw"))
    assert E._members(s.portfolio_key, 3) is E._members(s.portfolio_key, 3)
