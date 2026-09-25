"""Auswertung der Prognosekombination: sechs Mitglieder-Verfahren je Depot (das Boosting global), acht Kombinationsverfahren im Rolling-Origin-Vergleich über das Testjahr, dazu drei Experimente.

Kennzahl: **MASE** je Depot (MAE über alle Ursprünge des Testjahres und Horizonte, geteilt durch den saisonal naiven Trainingsfehler), über die Depots gemittelt. Referenzen: das beste Mitglied **im Nachhinein** je Depot
(nicht erreichbar, weil es erst nach dem Testjahr feststeht) und das beste Mitglied **insgesamt** (ebenfalls erst im Nachhinein bekannt)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import cmb_baselines as B
import cmb_combine as K
import cmb_constants as C
import cmb_members as M
import cmb_scenario as S


@dataclass(frozen=True)
class Settings:
    n_depots: int = C.DEFAULT_DEPOTS
    noise: float = C.DEFAULT_NOISE
    trend: int = C.DEFAULT_TREND
    swing: float = C.DEFAULT_SWING
    horizon: int = C.DEFAULT_HORIZON
    members: tuple = tuple(C.DEFAULT_MEMBERS)
    window: int = C.DEFAULT_WINDOW
    eta: float = C.DEFAULT_ETA
    seed: int = 3

    @property
    def portfolio_key(self):
        return (self.n_depots, self.noise, self.trend, self.swing, self.seed)


@dataclass
class Analysis:
    settings: Settings
    port: S.Portfolio
    test_org: np.ndarray
    actual: np.ndarray            # (n, T, h)
    members: dict                 # Mitglied -> (n, T, h)
    forecasts: dict               # Verfahren -> (n, T, h) (Kombinationen)
    weights: dict                 # Verfahren -> (n, T, K) oder None
    scale: np.ndarray
    mase: dict                    # Name -> (n,) MASE je Depot (Mitglieder und Kombinationen)
    summary: dict                 # Name -> MASE gemittelt über die Depots
    oracle_depot: float           # bestes Mitglied je Depot im Nachhinein
    best_member: str              # bestes Mitglied insgesamt (im Nachhinein)


@lru_cache(maxsize=16)
def _portfolio(key):
    n, noise, trend, swing, seed = key
    return S.generate(n, noise, 0.5, float(trend), seed, swing)


@lru_cache(maxsize=8)
def _members(key, horizon):
    """Alle sechs Mitglieder für alle Ursprünge ab FIT_END (teuer, deshalb je Portfolio und Horizont gemerkt)."""
    return M.member_forecasts(_portfolio(key), horizon)


def _scales(port):
    return np.array([B.mase_scale(port.y[i]) for i in range(port.n)])


def _mase(f, Y, scale):
    return np.abs(f - Y).mean(axis=(1, 2)) / scale


def assemble(port, s, member_fc):
    """Kombination und Kennzahlen aus den Mitglieder-Prognosen (dict Mitglied -> (n, O, h) ab FIT_END)."""
    h = s.horizon
    org = M.origins(h)
    test_org = np.arange(C.FIRST_TEST, C.N_DAYS - h + 1)
    names = list(s.members)
    F = np.stack([member_fc[m] for m in names])
    Y = port.y[:, org[:, None] + np.arange(h)[None, :]]
    fc, wts = K.combine(F, Y, test_org, h, s.window, s.eta)
    T0 = C.FIRST_TEST - C.FIT_END
    Yt = Y[:, T0:]
    scale = _scales(port)
    mase = {m: _mase(member_fc[m][:, T0:], Yt, scale) for m in names}
    mase.update({m: _mase(f, Yt, scale) for m, f in fc.items()})
    summ = {k: float(v.mean()) for k, v in mase.items()}
    stack = np.stack([mase[m] for m in names])
    best = min(names, key=lambda m: summ[m])
    return Analysis(s, port, test_org, Yt, {m: member_fc[m][:, T0:] for m in names}, fc, wts, scale, mase, summ, float(stack.min(axis=0).mean()), best)


@lru_cache(maxsize=8)
def analyse(s):
    port = _portfolio(s.portfolio_key)
    return assemble(port, s, {m: _members(s.portfolio_key, s.horizon)[m] for m in s.members})


# --- Aufschlüsselungen ---------------------------------------------------------------------------------------------------------------------------------

def mase_by_horizon(a):
    """MASE je Horizont für Mitglieder und Kombinationen: {Name: (h,)}."""
    sc = a.scale[:, None]
    allf = {**a.members, **a.forecasts}
    return {k: (np.abs(f - a.actual).mean(axis=1) / sc).mean(axis=0) for k, f in allf.items()}


def mase_by_target_day(a):
    """MASE je Zieltag im Testjahr (nur Tage, an denen alle Horizonte beitragen): {Name: (Tage,)}."""
    h = a.settings.horizon
    sc = a.scale[:, None, None]
    days = a.test_org[:, None] + np.arange(h)[None, :]
    out = {}
    for k, f in {**a.members, **a.forecasts}.items():
        err = np.abs(f - a.actual) / sc                                                    # (n, T, h)
        e = err.mean(axis=0)
        tot, cnt = np.zeros(C.N_DAYS), np.zeros(C.N_DAYS)
        np.add.at(tot, days.ravel(), e.ravel())
        np.add.at(cnt, days.ravel(), 1.0)
        out[k] = (tot / np.maximum(cnt, 1.0))[C.FIRST_TEST + h - 1:C.N_DAYS - h + 1]
    return out


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------------------------

def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def _replace(base, **kw):
    d = {f: getattr(base, f) for f in base.__dataclass_fields__}
    d.update(kw)
    return Settings(**d)


def pool_experiment(sizes=None, seeds=None, base=None):
    """Wächst der Pool, schlechte Mitglieder kommen hinzu: der Pool der Größe k enthält die ersten k Mitglieder der Reihenfolge Regression, Boosting, Holt-Winters, Wochenmittel, Letzte Woche, Naiv (vom besten zum schlechtesten
    im Vergleich ohne Schwankungen). Gezeigt: MASE je Kombinationsverfahren und bestes Mitglied des Pools."""
    sizes = C.POOL_SIZES if sizes is None else sizes
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    order = ("reg", "gbm", "hw", "wm", "snaive1", "naive")
    rows = []
    for k in sizes:
        per = {m: [] for m in C.METHODS}
        best = []
        for sd in seeds:
            a = analyse(_replace(base, members=order[:k], seed=sd))
            for m in C.METHODS:
                per[m].append(a.summary[m])
            best.append(min(a.summary[m] for m in order[:k]))
        rows.append({"k": k, "members": order[:k], "n_seeds": len(seeds), "methods": {m: _mean_se(v) for m, v in per.items()}, "best": _mean_se(best)})
    return rows


def swing_experiment(levels=None, seeds=None, base=None):
    """Wachsende depoteigene Niveauschwankung: die statische Regression verliert, die anpassungsfähigen Mitglieder gewinnen; MASE je Verfahren und Mitglied."""
    levels = C.SWING_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for b in levels:
        per = {}
        for sd in seeds:
            a = analyse(_replace(base, swing=b, seed=sd))
            for k, v in a.summary.items():
                per.setdefault(k, []).append(v)
            per.setdefault("_oracle", []).append(a.oracle_depot)
        rows.append({"swing": b, "n_seeds": len(seeds), "mase": {k: _mean_se(v) for k, v in per.items()}})
    return rows


def window_experiment(levels=None, seeds=None, base=None):
    """Kalibrierfenster für die geschätzten Gewichte: MASE der datengetriebenen Verfahren; wenig Daten bedeuten verrauschte Gewichte (das Kombinations-Rätsel)."""
    levels = C.WINDOW_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(n_depots=C.EXP_DEPOTS) if base is None else base
    rows = []
    for w in levels:
        per = {}
        for sd in seeds:
            a = analyse(_replace(base, window=w, seed=sd))
            for k, v in a.summary.items():
                per.setdefault(k, []).append(v)
        rows.append({"window": w, "n_seeds": len(seeds), "mase": {k: _mean_se(v) for k, v in per.items()}})
    return rows
