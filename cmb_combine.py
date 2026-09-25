"""Kombination von Prognosen: aus K Mitglieder-Prognosen je Depot, Ursprung und Horizont wird EINE Prognose. Alle Gewichte werden nur aus realisierten Fehlern geschätzt: zu einem Testursprung t zählen die Prognosen der Ursprünge o <= t - h (Zieltage vor t).

Fehler sind relativ: Ist und Prognosen werden durch s = mittlere Mitglieder-Prognose + 1 geteilt (sonst dominieren starke Wochentage die Gewichte).

  mean      Mittelwert der Mitglieder (alle Gewichte 1/K)
  median    Median der Mitglieder
  trim      Mittelwert ohne das kleinste und das größte Mitglied (erst ab vier Mitgliedern, sonst Mittelwert)
  invmse    Gewichte proportional zu 1 / MSE_k aus dem Kalibrierfenster der letzten W Ursprünge (Bates/Granger 1969)
  ols       Gewichte w = argmin sum (y - w'F)^2 im Fenster, frei (Granger/Ramanathan 1984), mit winziger Ridge-Strafe
  simplex   dasselbe mit w >= 0, sum w = 1 (exakt: alle Träger durchprobiert)
  ewa       Gewichte proportional zu D_k^(-eta) mit dem exponentiell abklingenden MSE D_k (Zerfall 1 - 1/W); eta = 0: Mittelwert, eta = 1: Inverse MSE (mit Zerfall), groß: Auswahl
  select    das Mitglied mit dem kleinsten MSE im Fenster (Modellauswahl statt Kombination)"""

import numpy as np

import cmb_constants as C


def _prep(F, Y):
    """Relative Größen: F (K, n, O, h), Y (n, O, h) -> Fr (n, O, h, K), Yr (n, O, h)."""
    s = F.mean(axis=0) + 1.0
    return np.moveaxis(F / s, 0, -1), Y / s, s


def _stats(Fr, Yr):
    """Je Ursprung über die Horizonte: A = sum F'F (n, O, K, K), b = sum F y (n, O, K), sq = sum (y - F_k)^2 (n, O, K)."""
    A = np.einsum("nohk,nohl->nokl", Fr, Fr)
    b = np.einsum("nohk,noh->nok", Fr, Yr)
    sq = ((Yr[..., None] - Fr) ** 2).sum(axis=2)
    return A, b, sq


def _window_sums(X, ends, W):
    """Summe von X (n, O, ...) über die Ursprungsindizes (e - W, e] für jedes e in ends; Rückgabe (n, len(ends), ...) und die Zahl der Ursprünge im Fenster."""
    z = np.zeros((X.shape[0], 1) + X.shape[2:])
    C_ = np.concatenate([z, np.cumsum(X, axis=1)], axis=1)
    hi = np.asarray(ends) + 1
    lo = np.maximum(hi - W, 0)
    return C_[:, hi] - C_[:, lo], hi - lo


def solve_simplex(A, b):
    """min w'Aw - 2 b'w über dem Simplex {w >= 0, sum w = 1}, stapelweise und EXAKT: alle Träger (Teilmengen der K Mitglieder) werden durchprobiert; je Träger löst das KKT-System
    [[2A_SS, 1], [1', 0]] (w, lam) = (2 b_S, 1) den Gleichungsfall, zulässig sind die Lösungen mit w_S >= 0, und unter ihnen zählt der kleinste Zielwert. A: (..., K, K), b: (..., K)."""
    K = b.shape[-1]
    best_val = np.full(b.shape[:-1], np.inf)
    best_w = np.zeros(b.shape)
    tr = np.trace(A, axis1=-2, axis2=-1)[..., None, None] / K
    for mask in range(1, 2 ** K):
        S = [k for k in range(K) if mask >> k & 1]
        m = len(S)
        Ass = A[..., S, :][..., :, S] + 1e-9 * tr * np.eye(m)
        M = np.zeros(b.shape[:-1] + (m + 1, m + 1))
        M[..., :m, :m] = 2.0 * Ass
        M[..., :m, m] = 1.0
        M[..., m, :m] = 1.0
        rhs = np.concatenate([2.0 * b[..., S], np.ones(b.shape[:-1] + (1,))], axis=-1)
        sol = np.linalg.solve(M + 1e-12 * np.eye(m + 1), rhs[..., None])[..., 0]
        w_s = sol[..., :m]
        ok = (w_s >= -1e-10).all(axis=-1)
        w_full = np.zeros(b.shape)
        w_full[..., S] = np.maximum(w_s, 0.0)
        val = np.einsum("...k,...kl,...l->...", w_full, A, w_full) - 2.0 * np.einsum("...k,...k->...", b, w_full)
        better = ok & (val < best_val)
        best_val = np.where(better, val, best_val)
        best_w = np.where(better[..., None], w_full, best_w)
    return best_w


def solve_free(A, b, ridge=1e-6):
    """Freie kleinste Quadrate w = (A + rho I)^-1 b, stapelweise; rho = ridge * mittlere Spur / K."""
    K = b.shape[-1]
    tr = np.trace(A, axis1=-2, axis2=-1)[..., None, None] / K
    return np.linalg.solve(A + ridge * tr * np.eye(K) + 1e-12 * np.eye(K), b[..., None])[..., 0]


def combine(F, Y, test_org, horizon, window=C.DEFAULT_WINDOW, eta=C.DEFAULT_ETA, methods=C.METHODS):
    """Kombinierte Prognosen für alle Testursprünge. F: (K, n, O, h) Mitglieder-Prognosen der Ursprünge ab FIT_END, Y: (n, O, h) Ist. Rückgabe ({Verfahren: (n, T, h)}, {Verfahren: Gewichte (n, T, K) oder None})."""
    K = F.shape[0]
    T0 = C.FIRST_TEST - C.FIT_END
    idx = test_org - C.FIT_END
    Ft = F[:, :, idx]                                                                       # (K, n, T, h)
    out, wts = {}, {}
    Fn = np.moveaxis(Ft, 0, -1)                                                             # (n, T, h, K)
    if "mean" in methods:
        out["mean"] = Fn.mean(axis=-1)
        wts["mean"] = np.full(Fn.shape[:2] + (K,), 1.0 / K)
    if "median" in methods:
        out["median"] = np.median(Fn, axis=-1)
        wts["median"] = None
    if "trim" in methods:
        nt = int(C.TRIM_SHARE * K)
        srt = np.sort(Fn, axis=-1)
        out["trim"] = srt[..., nt:K - nt].mean(axis=-1)
        wts["trim"] = None
    need = [m for m in ("invmse", "ols", "simplex", "ewa", "select") if m in methods]
    if need:
        Fr, Yr, _ = _prep(F, Y)
        A, b, sq = _stats(Fr, Yr)
        ends = idx - horizon                                                                # letzter Ursprung, dessen Zieltage vor t liegen
        Aw, cnt = _window_sums(A, ends, window)
        bw, _ = _window_sums(b, ends, window)
        sqw, _ = _window_sums(sq, ends, window)
        rows = (cnt * horizon)[None, :, None, None]
        mse = sqw / (cnt * horizon)[None, :, None]                                          # (n, T, K)
        if "invmse" in need:
            w = 1.0 / np.maximum(mse, 1e-12)
            wts["invmse"] = w / w.sum(axis=-1, keepdims=True)
        if "select" in need:
            wts["select"] = np.eye(K)[np.argmin(mse, axis=-1)]
        if "ols" in need:
            wts["ols"] = solve_free(Aw / rows, bw / rows[..., 0])
        if "simplex" in need:
            wts["simplex"] = solve_simplex(Aw / rows, bw / rows[..., 0])
        if "ewa" in need:
            lam = 1.0 - 1.0 / window
            Ls = sq / horizon                                                               # mittlerer Fehler je Ursprung (n, O, K)
            D = np.zeros_like(Ls)
            run = np.zeros(Ls.shape[:1] + Ls.shape[2:])
            for o in range(Ls.shape[1]):
                run = lam * run + Ls[:, o]
                D[:, o] = run
            Dt = np.maximum(D[:, ends], 1e-12)
            w = Dt ** (-eta)
            wts["ewa"] = w / w.sum(axis=-1, keepdims=True)
        for m in need:
            out[m] = np.maximum(np.einsum("nthk,ntk->nth", Fn, wts[m]), 0.0)
    order = [m for m in methods if m in out]
    return {m: out[m] for m in order}, {m: wts[m] for m in order}
