"""Die Kombinationsverfahren von Hand nachgerechnet, gegen unabhängige Schleifen und scipy (nur als Gegenprobe in requirements-dev), und ohne Blick in die Zukunft."""

import numpy as np
import pytest

import cmb_combine as K
import cmb_constants as C

H = 3
T_ORG = np.arange(C.FIRST_TEST, C.N_DAYS - H + 1)
O_EXT = C.N_DAYS - H + 1 - C.FIT_END


def _synthetic(n=2, k=3, seed=0, noise=(0.05, 0.10, 0.20)):
    """Mitglieder = Wahrheit * (1 + Fehler); Ist = Wahrheit * (1 + kleines Rauschen)."""
    rng = np.random.default_rng(seed)
    truth = 100.0 + 20 * rng.random((n, O_EXT, H))
    Y = truth * (1 + 0.02 * rng.normal(size=truth.shape))
    F = np.stack([truth * (1 + noise[i] * rng.normal(size=truth.shape)) for i in range(k)])
    return F, Y


def test_simplex_solver_by_hand():
    # A = I: die Projektion von b auf das Simplex; b = (0,7; 0,2; -0,3) -> (0,75; 0,25; 0)
    assert np.allclose(K.solve_simplex(np.eye(3), np.array([0.7, 0.2, -0.3])), [0.75, 0.25, 0.0])
    assert np.allclose(K.solve_simplex(np.eye(3), np.array([2.0, 0.0, 0.0])), [1, 0, 0])
    assert np.allclose(K.solve_simplex(np.eye(2), np.array([0.4, 0.4])), [0.5, 0.5])
    rng = np.random.default_rng(1)
    B = rng.normal(size=(30, 5, 5))
    A = np.einsum("nij,nkj->nik", B, B) / 5 + 0.1 * np.eye(5)
    w = K.solve_simplex(A, rng.normal(size=(30, 5)))
    assert (w >= 0).all() and np.allclose(w.sum(axis=-1), 1.0)


def test_simplex_solver_matches_scipy_and_the_kkt_conditions():
    opt = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(2)
    M = rng.normal(size=(4, 20))
    A = M @ M.T / 20
    b = A @ np.array([0.5, 0.3, 0.2, 0.0]) + 0.02 * rng.normal(size=4)
    w = K.solve_simplex(A, b)
    res = opt.minimize(lambda v: v @ A @ v - 2 * b @ v, np.full(4, 0.25), method="SLSQP", bounds=[(0, 1)] * 4, constraints={"type": "eq", "fun": lambda v: v.sum() - 1}, options={"ftol": 1e-14, "maxiter": 500})
    assert np.allclose(w, res.x, atol=1e-3) and (w >= 0).all() and w.sum() == pytest.approx(1.0)
    g = 2 * (A @ w - b)                                                                       # KKT: auf dem Träger gleicher Gradient, außerhalb nicht kleiner
    active = w > 1e-6
    assert np.ptp(g[active]) < 1e-3 and (g[~active] >= g[active].mean() - 1e-3).all()


def test_free_solver_matches_lstsq():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 4))
    y = X @ np.array([0.4, -0.2, 0.9, 0.1]) + 0.05 * rng.normal(size=300)
    w = K.solve_free(X.T @ X / 300, X.T @ y / 300, ridge=0.0)
    assert np.allclose(w, np.linalg.lstsq(X, y, rcond=None)[0], atol=1e-8)


def test_window_sums_by_loop():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(2, 50, 3))
    ends = np.array([10, 30, 49, 3])
    got, cnt = K._window_sums(X, ends, 12)
    for a, e in enumerate(ends):
        lo = max(e + 1 - 12, 0)
        assert np.allclose(got[:, a], X[:, lo:e + 1].sum(axis=1)) and cnt[a] == e + 1 - lo


def test_mean_median_trim_by_hand():
    F = np.zeros((5, 1, O_EXT, H))
    for k, v in enumerate([10.0, 20.0, 30.0, 40.0, 100.0]):
        F[k] = v
    Y = np.full((1, O_EXT, H), 25.0)
    out, w = K.combine(F, Y, T_ORG, H, methods=("mean", "median", "trim"))
    assert np.allclose(out["mean"], 40.0) and np.allclose(out["median"], 30.0) and np.allclose(out["trim"], 30.0)        # getrimmt: (20 + 30 + 40) / 3
    assert np.allclose(w["mean"], 0.2) and w["median"] is None
    out3, _ = K.combine(F[:3], Y, T_ORG, H, methods=("trim", "mean"))
    assert np.allclose(out3["trim"], out3["mean"])                                                                         # unter vier Mitgliedern: Mittelwert


def test_inverse_mse_and_selection_by_hand():
    # Mitglied 1 hat den Fehler 1 (relativ 1/s), Mitglied 2 den Fehler 2: MSE 1 : 4 -> Gewichte 0,8 : 0,2; Auswahl nimmt Mitglied 1
    truth = np.full((1, O_EXT, H), 100.0)
    F = np.stack([truth + 1.0, truth - 2.0])
    Y = truth.copy()
    out, w = K.combine(F, Y, T_ORG, H, window=30, methods=("invmse", "select"))
    s = F.mean(axis=0) + 1.0
    m1, m2 = (1.0 / s[0, 0, 0]) ** 2, (2.0 / s[0, 0, 0]) ** 2
    assert np.allclose(w["invmse"][0, 0], [(1 / m1) / (1 / m1 + 1 / m2), (1 / m2) / (1 / m1 + 1 / m2)]) and np.allclose(w["invmse"][0, 0], [0.8, 0.2])
    assert np.array_equal(w["select"][0, 0], [1.0, 0.0]) and np.allclose(out["select"], 101.0) and np.allclose(out["invmse"], 0.8 * 101 + 0.2 * 98)


def test_least_squares_recovers_the_true_weights():
    rng = np.random.default_rng(5)
    F = 100.0 + 30 * rng.random((2, 1, O_EXT, H))
    Y = 0.7 * F[0] + 0.3 * F[1]
    out, w = K.combine(F, Y, T_ORG, H, window=60, methods=("ols", "simplex"))
    assert np.allclose(w["ols"], [0.7, 0.3], atol=1e-4) and np.allclose(w["simplex"], [0.7, 0.3], atol=1e-4)
    assert np.allclose(out["ols"], Y[:, T_ORG - C.FIT_END], atol=0.1)


def test_simplex_weights_are_a_distribution_and_ols_can_leave_it():
    F, Y = _synthetic(n=3, k=3, seed=6)
    out, w = K.combine(F, Y, T_ORG, H, window=60, methods=K.C.METHODS)
    assert (w["simplex"] >= -1e-12).all() and np.allclose(w["simplex"].sum(axis=-1), 1.0)
    for m in ("invmse", "select", "ewa", "mean"):
        assert (w[m] >= 0).all() and np.allclose(w[m].sum(axis=-1), 1.0), m
    assert set(out) == set(C.METHODS) and all(v.shape == (3, len(T_ORG), H) for v in out.values())


def test_ewa_limits_and_discounted_loss_by_loop():
    F, Y = _synthetic(n=2, k=3, seed=7)
    W = 40
    out0, w0 = K.combine(F, Y, T_ORG, H, window=W, eta=0.0, methods=("ewa", "mean"))
    assert np.allclose(w0["ewa"], w0["mean"])
    out1, w1 = K.combine(F, Y, T_ORG, H, window=W, eta=1.0, methods=("ewa",))
    s = F.mean(axis=0) + 1.0
    Fr, Yr = F / s, Y / s
    lam = 1 - 1 / W
    i, t = 5, T_ORG[5]
    e = t - C.FIT_END - H
    for dep in range(2):
        D = np.zeros(3)
        for o in range(e + 1):
            D = lam * D + np.array([np.mean((Yr[dep, o] - Fr[k, dep, o]) ** 2) for k in range(3)]) * 1.0
        expect = (1 / D) / (1 / D).sum()
        assert np.allclose(w1["ewa"][dep, i], expect)


def test_only_realised_errors_are_used():
    F, Y = _synthetic(n=2, k=3, seed=8)
    base, wb = K.combine(F, Y, T_ORG, H, window=60)
    t = 800
    Y2 = Y.copy()
    Y2[:, t - C.FIT_END - H + 1:] = 999.0                          # alle Ursprünge, deren Zieltage nicht vor t liegen
    chg, wc = K.combine(F, Y2, T_ORG, H, window=60)
    i = t - C.FIRST_TEST
    for m in K.C.METHODS:
        assert np.allclose(base[m][:, :i + 1], chg[m][:, :i + 1]), m
    assert not np.allclose(base["invmse"], chg["invmse"])
    assert np.allclose(wb["simplex"][:, :i + 1], wc["simplex"][:, :i + 1])
