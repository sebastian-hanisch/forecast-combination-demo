"""Jede Zahl aus README und PRESET_HELP als Test. Die Verfahren sind deterministisch bei festem Seed; Bänder um die gerundeten Angaben, dazu Rangfolgen mit Abstand
(numpy-Versionen und Plattformen können den Zufallsstrom des HW-Fits und die Baumteilungen des Boostings um Rundung verschieben, feedback_ci_platform_robust_tests)."""

from functools import lru_cache

import numpy as np
import pytest

import cmb_constants as C
import cmb_evaluation as E
import cmb_presets as P

STD = "Standardfall: vier Mitglieder, 30 Depots"


@lru_cache(maxsize=None)
def _preset(name):
    v = P.PRESETS[name]
    return E.analyse(E.Settings(v["n_depots"], v["noise"], v["trend"], v["swing"], v["horizon"], tuple(v["members"]), v["window"], v["eta"], v["seed"]))


def _rel(a, m):
    return 100.0 * (a.summary[m] / a.summary[a.best_member] - 1.0)


def _wins(a):
    win = np.argmin(np.stack([a.mase[m] for m in a.members]), axis=0)
    return {m: int(np.sum(win == i)) for i, m in enumerate(a.members)}


def test_standard_preset():
    a = _preset(STD)
    s = a.summary
    assert a.best_member == "gbm" and s["gbm"] == pytest.approx(0.833, abs=0.03) and s["hw"] == pytest.approx(0.887, abs=0.03) and s["wm"] == pytest.approx(0.979, abs=0.04) and s["reg"] == pytest.approx(1.148, abs=0.08)
    assert _rel(a, "mean") == pytest.approx(1.0, abs=1.6) and _rel(a, "invmse") == pytest.approx(-0.8, abs=1.6) and _rel(a, "simplex") == pytest.approx(-1.1, abs=1.6) and _rel(a, "ewa") == pytest.approx(-2.2, abs=1.6) and _rel(a, "select") == pytest.approx(2.5, abs=1.6)
    assert s["ewa"] == pytest.approx(0.815, abs=0.03) and s["mean"] == pytest.approx(0.842, abs=0.03) and s["ewa"] < s["gbm"] < s["mean"] < s["select"] and _rel(a, "median") == pytest.approx(0.6, abs=1.6)
    w = _wins(a)
    assert w["gbm"] >= 16 and w["reg"] >= 3 and w["gbm"] + w["reg"] + w["hw"] + w["wm"] == 30 and w["wm"] == 0 and a.oracle_depot == pytest.approx(0.822, abs=0.03)


def test_no_swing_preset():
    a = _preset("Ohne Schwankung: die Regression ist die Wahrheit")
    s = a.summary
    assert a.best_member == "reg" and s["reg"] == pytest.approx(0.775, abs=0.03) and s["gbm"] == pytest.approx(0.790, abs=0.03)
    assert _rel(a, "ols") == pytest.approx(-4.0, abs=2) and _rel(a, "ewa") == pytest.approx(-3.8, abs=2) and _rel(a, "mean") == pytest.approx(0.7, abs=1.6) and _rel(a, "select") == pytest.approx(-2.1, abs=2)
    assert s["ols"] == pytest.approx(0.744, abs=0.03) and s["ewa"] == pytest.approx(0.745, abs=0.03) and s["mean"] == pytest.approx(0.780, abs=0.03) and s["select"] == pytest.approx(0.758, abs=0.03)
    assert s["ols"] < s["reg"] < s["mean"] and s["ewa"] < s["reg"]


def test_strong_swing_preset():
    a = _preset("Starke Schwankung (0,12)")
    s = a.summary
    assert a.best_member == "gbm" and s["reg"] == pytest.approx(1.712, abs=0.12) and s["gbm"] == pytest.approx(0.906, abs=0.04)
    assert _rel(a, "mean") == pytest.approx(5.2, abs=2) and _rel(a, "median") == pytest.approx(0.2, abs=1.6) and _rel(a, "ewa") == pytest.approx(-1.4, abs=1.6) and _rel(a, "ols") == pytest.approx(4.9, abs=2.5)
    assert s["mean"] == pytest.approx(0.953, abs=0.04) and s["median"] == pytest.approx(0.908, abs=0.04) and s["ewa"] == pytest.approx(0.894, abs=0.04) and s["ols"] == pytest.approx(0.950, abs=0.05)
    assert s["mean"] > s["gbm"] + 0.02 and s["median"] < s["mean"] and s["ewa"] < s["mean"]


def test_bad_member_preset():
    a = _preset("Ein schlechtes Mitglied im Pool (Naiv)")
    s = a.summary
    assert a.best_member == "gbm" and s["naive"] == pytest.approx(2.853, abs=0.15) and s["gbm"] == pytest.approx(0.833, abs=0.03)
    assert s["mean"] == pytest.approx(1.045, abs=0.05) and _rel(a, "mean") == pytest.approx(25.5, abs=5) and _rel(a, "median") == pytest.approx(2.4, abs=2) and _rel(a, "trim") == pytest.approx(2.1, abs=2)
    assert _rel(a, "ewa") == pytest.approx(-2.3, abs=1.6) and _rel(a, "simplex") == pytest.approx(-1.0, abs=1.6) and s["median"] == pytest.approx(0.853, abs=0.03) and s["trim"] == pytest.approx(0.850, abs=0.03)
    assert s["ewa"] == pytest.approx(0.814, abs=0.03) and s["simplex"] == pytest.approx(0.825, abs=0.03) and s["mean"] > 1.1 * s["median"]


def test_small_window_preset():
    a = _preset("Kleines Kalibrierfenster (30)")
    s = a.summary
    assert _rel(a, "ols") == pytest.approx(3.9, abs=2.5) and _rel(a, "simplex") == pytest.approx(-0.7, abs=1.6) and _rel(a, "ewa") == pytest.approx(-1.9, abs=1.6) and s["ols"] == pytest.approx(0.865, abs=0.04) and s["simplex"] == pytest.approx(0.827, abs=0.03)
    assert s["ewa"] == pytest.approx(0.817, abs=0.03) and s["mean"] == pytest.approx(0.842, abs=0.03) and s["ols"] > s["mean"] and s["ols"] > _preset(STD).summary["ols"] + 0.01


def test_two_member_preset():
    a = _preset("Nur Holt-Winters und Regression")
    s = a.summary
    assert a.best_member == "hw" and s["hw"] == pytest.approx(0.887, abs=0.03) and s["reg"] == pytest.approx(1.148, abs=0.08)
    assert _rel(a, "mean") == pytest.approx(1.2, abs=2) and _rel(a, "ols") == pytest.approx(-5.9, abs=2.5) and _rel(a, "ewa") == pytest.approx(-4.1, abs=2) and _rel(a, "invmse") == pytest.approx(-3.9, abs=2)
    assert s["ols"] == pytest.approx(0.834, abs=0.03) and s["ewa"] == pytest.approx(0.850, abs=0.03) and s["invmse"] == pytest.approx(0.852, abs=0.03) and s["mean"] == pytest.approx(0.897, abs=0.03)
    w = _wins(a)
    assert w["hw"] >= 12 and w["reg"] >= 6 and w["hw"] + w["reg"] == 30 and s["ols"] < s["hw"] and s["mean"] > s["hw"] - 0.01


@lru_cache(maxsize=None)
def _swing():
    return E.swing_experiment(C.SWING_LEVELS, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _pool():
    return E.pool_experiment(C.POOL_SIZES, C.EXP_SEEDS)


@lru_cache(maxsize=None)
def _window():
    return E.window_experiment(C.WINDOW_LEVELS, C.EXP_SEEDS)


def test_swing_experiment():
    rows = {r["swing"]: {k: v[0] for k, v in r["mase"].items()} for r in _swing()}
    r0, r6, r12 = rows[0.0], rows[0.06], rows[0.12]
    assert r0["reg"] == pytest.approx(0.744, abs=0.03) and r0["gbm"] == pytest.approx(0.804, abs=0.03) and r0["hw"] == pytest.approx(0.855, abs=0.03) and r0["wm"] == pytest.approx(0.941, abs=0.04)
    assert r0["mean"] == pytest.approx(0.780, abs=0.03) and r0["ols"] == pytest.approx(0.739, abs=0.03) and r0["simplex"] == pytest.approx(0.739, abs=0.03) and r0["ewa"] == pytest.approx(0.743, abs=0.03) and r0["select"] == pytest.approx(0.747, abs=0.03) and r0["_oracle"] == pytest.approx(0.742, abs=0.03)
    assert r6["reg"] == pytest.approx(1.011, abs=0.08) and r6["gbm"] == pytest.approx(0.835, abs=0.03) and r6["mean"] == pytest.approx(0.826, abs=0.03) and r6["ewa"] == pytest.approx(0.804, abs=0.03) and r6["select"] == pytest.approx(0.850, abs=0.03) and r6["simplex"] == pytest.approx(0.806, abs=0.03)
    assert r12["reg"] == pytest.approx(1.456, abs=0.12) and r12["gbm"] == pytest.approx(0.890, abs=0.04) and r12["mean"] == pytest.approx(0.915, abs=0.04) and r12["median"] == pytest.approx(0.884, abs=0.04) and r12["ewa"] == pytest.approx(0.871, abs=0.04) and r12["select"] == pytest.approx(0.922, abs=0.04)
    assert r0["median"] == pytest.approx(0.781, abs=0.03) and r6["median"] == pytest.approx(0.826, abs=0.03) and r6["ols"] == pytest.approx(0.811, abs=0.03)
    assert r0["mean"] > r0["reg"] and r0["ols"] < r0["reg"] + 0.01 and r12["mean"] > r12["gbm"] and r12["ewa"] < r12["gbm"] + 0.005 and r12["select"] > r12["ewa"] and r0["reg"] < r0["gbm"] < r12["gbm"]


def test_pool_experiment():
    rows = {r["k"]: r for r in _pool()}
    m = lambda k, name: rows[k]["methods"][name][0]
    assert rows[2]["best"][0] == pytest.approx(0.835, abs=0.03) and rows[6]["best"][0] == pytest.approx(0.835, abs=0.03)
    assert m(2, "mean") == pytest.approx(0.843, abs=0.03) and m(3, "mean") == pytest.approx(0.816, abs=0.03) and m(4, "mean") == pytest.approx(0.826, abs=0.03) and m(5, "mean") == pytest.approx(0.850, abs=0.03) and m(6, "mean") == pytest.approx(0.982, abs=0.05)
    assert m(6, "median") == pytest.approx(0.855, abs=0.03) and m(6, "simplex") == pytest.approx(0.807, abs=0.03) and m(6, "ewa") == pytest.approx(0.804, abs=0.03) and m(6, "invmse") == pytest.approx(0.828, abs=0.03) and m(6, "ols") == pytest.approx(0.815, abs=0.03)
    assert m(2, "ewa") == pytest.approx(0.804, abs=0.03) and m(2, "ols") == pytest.approx(0.805, abs=0.03) and m(4, "ewa") == pytest.approx(0.804, abs=0.03) and m(6, "select") == pytest.approx(0.848, abs=0.03)
    assert m(6, "mean") > m(2, "mean") + 0.1 and m(6, "mean") > m(6, "median") + 0.09 and abs(m(6, "simplex") - m(2, "simplex")) < 0.03 and abs(m(6, "ewa") - m(2, "ewa")) < 0.03 and m(6, "mean") > rows[6]["best"][0] + 0.1


def test_window_experiment():
    rows = {r["window"]: {k: v[0] for k, v in r["mase"].items()} for r in _window()}
    assert rows[30]["ols"] == pytest.approx(0.850, abs=0.04) and rows[60]["ols"] == pytest.approx(0.822, abs=0.04) and rows[120]["ols"] == pytest.approx(0.805, abs=0.03)
    assert rows[30]["simplex"] == pytest.approx(0.812, abs=0.03) and rows[120]["simplex"] == pytest.approx(0.806, abs=0.03) and rows[30]["ewa"] == pytest.approx(0.807, abs=0.03) and rows[120]["ewa"] == pytest.approx(0.803, abs=0.03)
    assert rows[60]["simplex"] == pytest.approx(0.807, abs=0.03) and rows[60]["ewa"] == pytest.approx(0.805, abs=0.03)
    assert rows[30]["invmse"] == pytest.approx(0.811, abs=0.03) and rows[30]["select"] == pytest.approx(0.845, abs=0.03) and rows[30]["mean"] == pytest.approx(0.826, abs=0.03)
    assert rows[30]["mean"] == pytest.approx(rows[120]["mean"], abs=1e-9) and rows[30]["ols"] > rows[30]["mean"] + 0.01 and rows[30]["ols"] > rows[120]["ols"] + 0.02 and abs(rows[30]["simplex"] - rows[120]["simplex"]) < 0.02
