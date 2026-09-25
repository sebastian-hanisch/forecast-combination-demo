"""Die Mitglieder des Pools: sechs Prognoseverfahren aus den Vorgängern, je Depot (das Boosting global über alle Depots), alle mit Parametern aus den Tagen vor FIT_END.

Ursprung t: bekannt sind die Tage 0..t-1, prognostiziert werden t..t+h-1. Die Prognosen werden für alle Ursprünge ab FIT_END berechnet: die Tage FIT_END..FIRST_TEST-1 sind das Kalibrierfenster für die Gewichte, ab FIRST_TEST beginnt das Testjahr."""

import numpy as np

import cmb_baselines as B
import cmb_constants as C
import cmb_features as F
import cmb_gbm as G


def origins(horizon):
    return np.arange(C.FIT_END, C.N_DAYS - horizon + 1)


def naive(y, org, h):
    """Der letzte bekannte Wert für alle Horizonte: (n_org, h)."""
    return np.repeat(y[org - 1][:, None], h, axis=1)


def gbm_forecasts(port, horizon):
    """Ein globales Boosting-Modell über alle Depots (Stück 6): Zieltage vor FIT_END im Training, Ziel = log-Verhältnis zum 28-Tage-Niveau; Rückgabe (n, n_org, h)."""
    depots = np.arange(port.n)
    d, o, h = F.training_rows(port, depots, horizon, stride=4, per_origin=2, last_target=C.FIT_END, seed=0)
    X, y, _ = F.build(port, d, o, h)
    ens = G.fit(X, y, num_leaves=15, min_child_samples=20, n_rounds=C.GBM_ROUNDS, learning_rate=0.1, seed=0)
    td, to, th, org = F.test_rows(port, depots, horizon, first=C.FIT_END)
    Xt, _, lvl = F.build(port, td, to, th)
    pred = F.to_orders(G.predict(ens, Xt), lvl)
    return pred.reshape(port.n, len(org), horizon)


def member_forecasts(port, horizon, members=C.MEMBERS):
    """{Mitglied: (n, n_org, h)} für alle Ursprünge ab FIT_END."""
    org = origins(horizon)
    out = {m: [] for m in members if m != "gbm"}
    for i in range(port.n):
        y = port.y[i]
        for m in out:
            if m == "naive":
                out[m].append(naive(y, org, horizon))
            elif m == "snaive1":
                out[m].append(B.snaive_k(y, org, horizon, k=1))
            elif m == "wm":
                out[m].append(B.snaive_k(y, org, horizon, k=4))
            elif m == "hw":
                out[m].append(B.hw_forecast(y, org, horizon, first=C.FIT_END))
            elif m == "reg":
                X = B.regression_design(port.dow, port.holiday, port.after, port.promo[i])
                out[m].append(B.regression_forecast(y, X, org, horizon, first=C.FIT_END))
    res = {m: np.stack(v) for m, v in out.items()}
    if "gbm" in members:
        res["gbm"] = gbm_forecasts(port, horizon)
    return {m: res[m] for m in members}
