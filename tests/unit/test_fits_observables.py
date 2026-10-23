"""Level-0 tests: decay fits, storage round trip, thermal band temperature."""
import tempfile

import numpy as np
import pandas as pd

from rbdsmc.constants import KB
from rbdsmc.diagnostics.fits import (analyse_decay, fit_late_exponential, fit_one_two_body,
                                     fit_transient, one_two_body, time_to_fraction)
from rbdsmc.diagnostics.observables import thermal_band_temperature
from rbdsmc.io.storage import RunWriter, read_table


def test_time_to_fraction():
    t = np.linspace(0, 10, 1001)
    assert abs(time_to_fraction(t, np.exp(-t / 2.0)) - 2.0) < 1e-3
    assert np.isnan(time_to_fraction(t, np.ones_like(t)))


def test_transient_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 200, 401)
    f = 0.15 + 0.85 * np.exp(-t / 8.0) + rng.normal(0, 0.002, t.size)
    r = fit_transient(t, f)
    assert abs(r["f_plateau"] - 0.15) < 0.003 and abs(r["tau_transient"] / 8.0 - 1) < 0.03


def test_late_exponential_and_flat():
    t = np.linspace(0, 200, 201)
    r = fit_late_exponential(t, 1e5 * np.exp(-t / 150.0), 60)
    assert abs(r["tau_late"] / 150 - 1) < 1e-6
    rng = np.random.default_rng(1)
    flat = 1e4 * (1 + rng.normal(0, 1e-3, t.size))
    assert np.isinf(fit_late_exponential(t, flat, 60)["tau_late"])


def test_one_two_body_fit():
    t = np.linspace(0, 200, 201)
    N = one_two_body(t, 1e5, 300.0, 2e-7)
    r = fit_one_two_body(t, N, 0.0)
    assert abs(r["tau1"] / 300 - 1) < 1e-3 and abs(r["beta"] / 2e-7 - 1) < 1e-3
    # ODE check
    dNdt = np.gradient(N, t)
    assert np.allclose(dNdt[5:-5], -N[5:-5] / 300 - 2e-7 * N[5:-5] ** 2, rtol=1e-3)


def test_analyse_decay_runs_on_plateau():
    t = np.linspace(0, 200, 401)
    N = 1e5 * (0.15 + 0.85 * np.exp(-t / 6.0))
    r = analyse_decay(t, N)
    assert abs(r["f_plateau"] - 0.15) < 1e-3
    assert r["t_1e"] < 10 and (np.isinf(r["tau_late"]) or r["tau_late"] > 1e4)


def test_thermal_band_temperature_harmonic_ladder():
    hw = 2.3e-6 * KB
    eps = hw * np.arange(30)
    T = 4e-6
    w = np.exp(-eps / (KB * T))
    target = np.sum(eps * w) / np.sum(w)
    assert abs(thermal_band_temperature(eps, target) / T - 1) < 1e-6
    assert np.isnan(thermal_band_temperature(eps[:3], 10 * hw))


def test_storage_roundtrip_both_backends():
    df = pd.DataFrame({"t_ms": [0.0, 0.5], "f": [1.0, 0.9]})
    for backend in ("csv", "parquet"):
        try:
            if backend == "parquet":
                import pyarrow  # noqa: F401
        except ImportError:
            continue
        with tempfile.TemporaryDirectory() as d:
            w = RunWriter(d, backend=backend)
            w.table("timeseries", df)
            w.json("summary", {"a": np.float64(1.5), "b": np.inf, "c": np.arange(3)})
            back = read_table(d, "timeseries")
            assert np.allclose(back.values, df.values)
