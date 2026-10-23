"""Decay-time analysis of the atom number in the trap region.

Reported quantities (times in the units of ``t``):
  t_1e        first time N falls to N(0)/e (linear interpolation); nan if never.
  transient   model-free: f_plateau = mean of the last 10% of the record; tau_transient =
              first time the excess f - f_plateau falls to (1 - f_plateau)/e.  The prompt
              loss of unbound atoms is ballistic (delayed), not exponential, so no fit.
              (``fit_transient`` keeps the exponential fit for reference.)
  late        log-linear fit over t >= t_late: tau_late (inf when no measurable decay).
  one+two     fit  dN/dt = -N/tau1 - beta N^2  over t >= t_late
              (analytic solution); tau1, beta.  Meaningful with collisions.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def time_to_fraction(t, N, level=np.exp(-1.0)):
    t, N = np.asarray(t, float), np.asarray(N, float)
    target = level * N[0]
    below = np.nonzero(N <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (target - N[i - 1]) * (t[i] - t[i - 1]) / (N[i] - N[i - 1]))


def fit_transient(t, f):
    t, f = np.asarray(t, float), np.asarray(f, float)
    model = lambda tt, finf, A, tau: finf + A * np.exp(-tt / tau)    # noqa: E731
    f_end = f[-1]
    A0 = f[0] - f_end
    # initial tau: time to lose half of the eventual loss
    half = f_end + 0.5 * A0
    k = np.argmax(f <= half) if A0 > 0 else len(t) // 2
    tau0 = max(t[k] - t[0], (t[1] - t[0]) if len(t) > 1 else 1.0)
    try:
        p, cov = curve_fit(model, t, f, p0=[f_end, A0, tau0],
                           bounds=([-1, -2, 1e-12], [2, 2, 1e12]), maxfev=20000)
        err = np.sqrt(np.diag(cov))
        return {"f_plateau": p[0], "f_plateau_err": err[0], "amplitude": p[1],
                "tau_transient": p[2], "tau_transient_err": err[2]}
    except (RuntimeError, ValueError):
        return {"f_plateau": np.nan, "f_plateau_err": np.nan, "amplitude": np.nan,
                "tau_transient": np.nan, "tau_transient_err": np.nan}


def fit_late_exponential(t, N, t_start):
    t, N = np.asarray(t, float), np.asarray(N, float)
    m = (t >= t_start) & (N > 0)
    if m.sum() < 3:
        return {"tau_late": np.nan, "tau_late_err": np.nan, "late_slope": np.nan}
    (slope, icpt), cov = np.polyfit(t[m], np.log(N[m]), 1, cov=True)
    s_err = np.sqrt(cov[0, 0])
    span = t[m][-1] - t[m][0]
    # no measurable decay: non-negative or insignificant slope, or tau > 100 x the window
    if slope >= 0 or abs(slope) < 2 * s_err or -1.0 / slope > 100 * span:
        tau, tau_err = np.inf, np.nan
    else:
        tau, tau_err = -1.0 / slope, s_err / slope ** 2
    return {"tau_late": tau, "tau_late_err": tau_err, "late_slope": slope}


def one_two_body(t, N0, tau1, beta, t0=0.0):
    """Solution of dN/dt = -N/tau1 - beta N^2 with N(t0) = N0."""
    b = beta * N0 * tau1
    return N0 / ((1.0 + b) * np.exp((t - t0) / tau1) - b)


def fit_one_two_body(t, N, t_start):
    t, N = np.asarray(t, float), np.asarray(N, float)
    m = t >= t_start
    if m.sum() < 4 or N[m][0] <= 0:
        return {"tau1": np.nan, "tau1_err": np.nan, "beta": np.nan, "beta_err": np.nan}
    tt, NN = t[m], N[m]
    t0, N0 = tt[0], NN[0]
    scale = N0
    f = lambda x, n0, tau1, bb: one_two_body(x, n0, tau1, bb, t0)    # noqa: E731
    T_span = tt[-1] - t0
    try:
        p, cov = curve_fit(lambda x, n0, tau1, bb: f(x, n0 * scale, tau1, bb / scale) / scale,
                           tt, NN / scale, p0=[1.0, 10 * T_span, 1e-3 / T_span],
                           bounds=([0.0, 1e-6 * T_span, 0.0], [10.0, 1e9 * T_span, 1e6 / T_span]),
                           maxfev=20000)
        err = np.sqrt(np.diag(cov))
        return {"tau1": p[1], "tau1_err": err[1], "beta": p[2] / scale, "beta_err": err[2] / scale}
    except (RuntimeError, ValueError):
        return {"tau1": np.nan, "tau1_err": np.nan, "beta": np.nan, "beta_err": np.nan}


def transient_model_free(t, f, tail_fraction=0.1):
    t, f = np.asarray(t, float), np.asarray(f, float)
    n_tail = max(3, int(round(tail_fraction * len(t))))
    f_pl = float(np.mean(f[-n_tail:]))
    f_pl_err = float(np.std(f[-n_tail:]) / np.sqrt(n_tail))
    excess = f - f_pl
    tau = time_to_fraction(t, np.maximum(excess, 0.0) + 1e-300, level=np.exp(-1.0)) \
        if excess[0] > 0 else np.nan
    return {"f_plateau": f_pl, "f_plateau_err": f_pl_err, "tau_transient": tau}


def analyse_decay(t, N, late_fraction=0.3):
    """All decay quantities for N(t) (any units); late fits use t >= late_fraction * t_end."""
    t, N = np.asarray(t, float), np.asarray(N, float)
    f = N / N[0]
    out = {"t_1e": time_to_fraction(t, N)}
    out.update(transient_model_free(t, f))
    t_late = t[0] + late_fraction * (t[-1] - t[0])
    out["t_late_start"] = t_late
    out.update(fit_late_exponential(t, N, t_late))
    out.update(fit_one_two_body(t, N, t_late))
    return out
