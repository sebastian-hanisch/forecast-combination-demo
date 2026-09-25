"""Plotly-Abbildungen der Kombinations-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go

import cmb_constants as C
import cmb_evaluation as E

MEMBER_COLORS = {"naive": "#9e9e9e", "snaive1": "#bcbd22", "wm": "#17becf", "hw": "#e6550d", "reg": "#8c6bb1", "gbm": "#2e7d32"}
METHOD_COLORS = {"mean": "#1f77b4", "median": "#7f7f7f", "trim": "#8c564b", "invmse": "#d62728", "ols": "#e377c2", "simplex": "#00897b", "ewa": "#0d47a1", "select": "#ff7f0e"}
ACTUAL = "#14233B"
WARN = "#f58518"
SHORT = {**C.MEMBER_SHORT, **C.METHOD_SHORT}
COLORS = {**MEMBER_COLORS, **METHOD_COLORS}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def build_series(a, dep, origin, shown):
    """42 Tage vor dem Ursprung und die nächsten h Tage: Ist, die Mitglieder (dünn) und die gewählten Kombinationen (dick)."""
    port, h = a.port, a.settings.horizon
    i = int(origin - a.test_org[0])
    x_hist, x_fut = np.arange(origin - 42, origin), np.arange(origin, origin + h)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_hist, y=port.y[dep, origin - 42:origin], mode="lines+markers", name="bekannt", line=dict(color=ACTUAL, width=1.5), marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=x_fut, y=port.y[dep, origin:origin + h], mode="lines+markers", name="tatsächlich", line=dict(color=ACTUAL, width=1), marker=dict(size=7, symbol="circle-open")))
    for m in a.members:
        fig.add_trace(go.Scatter(x=x_fut, y=a.members[m][dep, i], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=1.3, dash="dot")))
    for m in shown:
        fig.add_trace(go.Scatter(x=x_fut, y=a.forecasts[m][dep, i], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=3)))
    fig.add_vline(x=origin - 0.5, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Tag")
    fig.update_yaxes(title_text="Aufträge je Tag", rangemode="tozero")
    return _base(fig, 360).update_layout(legend=dict(orientation="h", y=-0.3))


def build_mase(a):
    """MASE der Mitglieder und der Kombinationen; Linien: bestes Mitglied im Nachhinein (insgesamt) und bestes Mitglied je Depot im Nachhinein."""
    names = list(a.members) + list(a.forecasts)
    vals = [a.summary[k] for k in names]
    fig = go.Figure(go.Bar(x=[SHORT[k] for k in names], y=vals, marker=dict(color=[COLORS[k] for k in names]), text=[de(v) for v in vals], textposition="outside", showlegend=False))
    fig.add_hline(y=a.summary[a.best_member], line=dict(color="#7f7f7f", dash="dash"), annotation_text=f"bestes Mitglied im Nachhinein: {SHORT[a.best_member]}", annotation_position="top left")
    fig.add_hline(y=a.oracle_depot, line=dict(color="#54a24b", dash="dot"), annotation_text="bestes Mitglied je Depot im Nachhinein", annotation_position="bottom left")
    fig.update_yaxes(title_text="MASE (kleiner ist besser)", rangemode="tozero")
    return _base(fig, 380)


def build_best_counts(a):
    """In wie vielen Depots ist welches Mitglied das beste (im Nachhinein)?"""
    names = list(a.members)
    stack = np.stack([a.mase[m] for m in names])
    win = np.argmin(stack, axis=0)
    counts = [int(np.sum(win == k)) for k in range(len(names))]
    fig = go.Figure(go.Bar(x=[SHORT[m] for m in names], y=counts, marker=dict(color=[COLORS[m] for m in names]), text=counts, textposition="outside", showlegend=False))
    fig.update_yaxes(title_text="Zahl der Depots, in denen das Mitglied am besten ist", rangemode="tozero")
    return _base(fig, 300)


def build_horizon(a, shown):
    h = a.settings.horizon
    vals = E.mase_by_horizon(a)
    xs = list(range(1, h + 1))
    fig = go.Figure()
    for m in a.members:
        fig.add_trace(go.Scatter(x=xs, y=vals[m], mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=1.3, dash="dot")))
    for m in shown:
        fig.add_trace(go.Scatter(x=xs, y=vals[m], mode="lines+markers", name=SHORT[m], line=dict(color=COLORS[m], width=2.6), marker=dict(size=5)))
    fig.update_xaxes(title_text="Prognosehorizont (Tage)", dtick=1 if h <= 14 else 2)
    fig.update_yaxes(title_text="MASE je Horizont", rangemode="tozero")
    return _base(fig, 320)


def _rolling(v, w=21):
    k = np.ones(w) / w
    pad = np.concatenate([np.full(w // 2, v[0]), v, np.full(w // 2, v[-1])])
    return np.convolve(pad, k, mode="valid")[:len(v)]


def build_time(a, shown):
    h = a.settings.horizon
    vals = E.mase_by_target_day(a)
    x = np.arange(C.FIRST_TEST + h - 1, C.N_DAYS - h + 1)
    fig = go.Figure()
    for m in a.members:
        fig.add_trace(go.Scatter(x=x, y=_rolling(vals[m]), mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=1.3, dash="dot")))
    for m in shown:
        fig.add_trace(go.Scatter(x=x, y=_rolling(vals[m]), mode="lines", name=SHORT[m], line=dict(color=COLORS[m], width=2.6)))
    fig.update_xaxes(title_text="Zieltag")
    fig.update_yaxes(title_text="MASE (21-Tage-Mittel)", rangemode="tozero")
    return _base(fig, 320)


def build_weights(a, dep, method):
    """Gewichte der Mitglieder über die Testursprünge für ein Depot (gestapelt); nur für Verfahren mit Gewichten."""
    w = a.weights[method]
    names = list(a.members)
    x = a.test_org
    fig = go.Figure()
    stacked = method != "ols"
    for k, m in enumerate(names):
        fig.add_trace(go.Scatter(x=x, y=w[dep, :, k], mode="lines", name=SHORT[m], stackgroup="w" if stacked else None, line=dict(color=COLORS[m], width=0.8 if stacked else 1.8)))
    fig.update_xaxes(title_text="Ursprung (Tag)")
    fig.update_yaxes(title_text=f"Gewicht ({SHORT[method]})")
    return _base(fig, 320)


# --- Experimente ------------------------------------------------------------------------------------------------------------------------------


def _lines(rows, xkey, xfmt, series, title_x, title_y="MASE (kleiner ist besser)", height=380, dash=None):
    xs = [xfmt(r[xkey]) for r in rows]
    fig = go.Figure()
    for name, getter, kind in series:
        y = [getter(r)[0] for r in rows]
        err = [getter(r)[1] for r in rows]
        fig.add_trace(go.Scatter(x=xs, y=y, error_y=dict(type="data", array=err), mode="lines+markers", name=SHORT.get(name, name), line=dict(color=COLORS.get(name, "#54a24b"), width=2.8 if kind == "method" else 1.6, dash="solid" if kind == "method" else "dot")))
    fig.update_xaxes(title_text=title_x, type="category")
    fig.update_yaxes(title_text=title_y, rangemode="tozero")
    return _base(fig, height).update_layout(legend=dict(orientation="h", y=-0.3))


def build_swing(rows, members):
    series = [(m, (lambda r, m=m: r["mase"][m]), "member") for m in members] + [(m, (lambda r, m=m: r["mase"][m]), "method") for m in ("mean", "median", "simplex", "ewa", "select")]
    return _lines(rows, "swing", lambda x: de(x, 2), series, "Niveauschwankung der Depots")


def build_pool(rows):
    series = [(m, (lambda r, m=m: r["methods"][m]), "method") for m in ("mean", "median", "trim", "invmse", "simplex", "ewa", "select")] + [("bestes Mitglied", (lambda r: r["best"]), "member")]
    labels = lambda k: str(k)
    fig = _lines(rows, "k", labels, series, "Mitglieder im Pool (vom besten zum schlechtesten hinzugefügt)")
    return fig


def build_window(rows, methods=("mean", "invmse", "ols", "simplex", "ewa", "select")):
    series = [(m, (lambda r, m=m: r["mase"][m]), "method") for m in methods]
    return _lines(rows, "window", lambda w: f"{w} Tage", series, "Kalibrierfenster für die Gewichte")
