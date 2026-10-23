"""Level-1/3 tests: golden-rule normalisation and the DSMC collider.

The collider tests use a 'uniform quasi-2D gas': atoms spread uniformly over one cell
group (K sites, a square of side L), with the local depth s forced to a constant and
positions redrawn every step (no propagation), so that rates can be compared with
analytic values.
"""
import numpy as np

from rbdsmc.config import load_config
from rbdsmc.constants import A_SCAT, HBAR, KB, M_RB
from rbdsmc.dynamics.dsmc import Collider
from rbdsmc.diagnostics.observables import thermal_band_temperature
from rbdsmc.physics.potentials import SemiclassicalTrap
from rbdsmc.physics.scattering import (OverlapTable, S_BOSE, coupling_constant,
                                       harmonic_overlaps, thermal_rate_quasi2d)
from rbdsmc.state import ParticleState, Status

_C = {}


def env():
    if not _C:
        cfg = load_config(None, overrides=["physics.collisions=true"])
        T = SemiclassicalTrap(cfg)
        tab = OverlapTable.load_or_compute("results/cache")
        _C.update(cfg=cfg, T=T, tab=tab)
    return _C["cfg"], _C["T"], _C["tab"]


def test_quasi2d_rate_tends_to_3d_limit():
    w = 2 * np.pi * 48e3
    l = np.sqrt(HBAR / (M_RB * w))
    hw = HBAR * w
    C = S_BOSE * coupling_constant() ** 2 * (M_RB / 2) / HBAR ** 3
    ratios = []
    for nmax, r in [(30, 1.0), (30, 2.0), (40, 4.0)]:
        I = harmonic_overlaps(nmax, l)
        kT = r * hw
        G2 = thermal_rate_quasi2d(hw * (np.arange(nmax) + 0.5), I, kT, 1e10, C)
        sz2 = HBAR / (2 * M_RB * w) / np.tanh(hw / (2 * kT))
        G3 = 1e10 / (2 * np.sqrt(np.pi * sz2)) * 8 * np.pi * A_SCAT ** 2 * np.sqrt(16 * kT / (np.pi * M_RB))
        ratios.append(G2 / G3)
    assert ratios[0] > ratios[1] > ratios[2] > 1.0       # monotone approach from above
    assert ratios[2] < 1.04


def test_harmonic_overlap_values():
    I = harmonic_overlaps(4, 1.0)
    I0 = 1 / np.sqrt(2 * np.pi)
    assert np.isclose(I[0, 0, 0, 0], I0) and np.isclose(I[0, 0, 1, 1], I0 / 2)
    assert np.isclose(I[1, 1, 1, 1], 3 * I0 / 4) and abs(I[0, 0, 0, 1]) < 1e-14


def _uniform_gas(T, K, N, s0, kT, rng, n_init=None, L=100e-6):
    st = ParticleState.empty(N)
    st.status[:] = Status.BOUND
    G = T.j_c // K
    st.j[:] = rng.integers(G * K, (G + 1) * K, N)
    st.n[:] = 0 if n_init is None else n_init
    st.s[:] = s0
    st.z[:] = T.z_site_c
    sig = np.sqrt(kT / M_RB)
    st.vx[:], st.vy[:] = rng.normal(0, sig, N), rng.normal(0, sig, N)
    return st, L


def _redraw_positions(T, st, L, rng, col):
    """Uniform positions in a square whose edges lie on cell boundaries (no partial cells)."""
    N = st.N
    x0 = -col.R + col.h * np.floor((col.R - L / 2) / col.h)
    st.x[:] = T.x_center(T.z_c) + x0 + rng.uniform(0, L, N)
    st.y[:] = x0 + rng.uniform(0, L, N)


def _collide(col, st, T, L, rng, nsteps, s0):
    n = 0
    for k in range(nsteps):
        _redraw_positions(T, st, L, rng, col)
        st.s[:] = s0
        n += col.step(st, k * col.dt)
    return n


def test_elastic_rate_matches_golden_rule():
    cfg, T, tab = env()
    rng = np.random.default_rng(0)
    s0, kT = 140.0, KB * 0.3e-6          # only the 00 -> 00 channel is open
    N, dt, Neff = 4000, 25e-6, 400.0
    col = Collider(T, cfg, rng, N_eff=Neff, dt=dt, overlaps=tab)
    st, L = _uniform_gas(T, col.K, N, s0, kT, rng)
    nsteps = 400
    ncoll = _collide(col, st, T, L, rng, nsteps, s0)
    rate_sim = 2 * ncoll / N / (nsteps * dt)
    n2d = N * Neff / (col.K * L ** 2)
    I0 = tab.I[:, 0, 0, 0, 0]
    I0s = np.interp(np.log(s0), np.log(tab.s_grid), I0)
    rate_th = n2d * col.C * I0s ** 2
    assert abs(rate_sim / rate_th - 1) < 0.05, (rate_sim, rate_th)
    assert np.all(st.n == 0)
    assert col.stats["violations"] == 0


def test_conservation_and_parity_with_band_changes():
    cfg, T, tab = env()
    rng = np.random.default_rng(1)
    s0, kT = 400.0, KB * 6e-6
    N, dt, Neff = 3000, 25e-6, 2000.0
    col = Collider(T, cfg, rng, N_eff=Neff, dt=dt, overlaps=tab)
    n0 = rng.integers(0, 3, N)
    st, L = _uniform_gas(T, col.K, N, s0, kT, rng, n_init=n0)

    def energy_momentum():
        # bound: 1/2 m v_perp^2 + E_band(s0);  released (FREE): 1/2 m v^2 - U_lat/2 (same s0)
        b = st.status == Status.BOUND
        f = st.status == Status.FREE
        E = (0.5 * M_RB * (st.vx[b] ** 2 + st.vy[b] ** 2)).sum() \
            + (T.ER * T.tab.energy(np.full(b.sum(), s0), st.n[b])).sum() \
            + (0.5 * M_RB * (st.vx[f] ** 2 + st.vy[f] ** 2 + st.vz[f] ** 2) - 0.5 * s0 * T.ER).sum()
        return E, st.vx.sum(), st.vy.sum(), int(st.n.sum() % 2)

    E0, px0, py0, par0 = energy_momentum()
    ncoll = _collide(col, st, T, L, rng, 200, s0)
    assert ncoll > 1000 and col.stats["band_changing"] > 0
    assert col.stats["released"] < 0.01 * N              # rare promotions above the barrier
    E1, px1, py1, par1 = energy_momentum()
    assert abs(E1 - E0) < 1e-9 * abs(E0)
    vscale = np.sqrt(kT / M_RB) * np.sqrt(N)
    assert abs(px1 - px0) < 1e-9 * vscale and abs(py1 - py0) < 1e-9 * vscale
    assert par1 == par0                                  # parity selection rule


def test_cross_dimensional_thermalisation():
    """Hot radial / cold axial gas relaxes to T_perp = T_par with energy conserved."""
    cfg, T, tab = env()
    rng = np.random.default_rng(2)
    s0 = 400.0
    N, dt, Neff = 3000, 25e-6, 4000.0
    col = Collider(T, cfg, rng, N_eff=Neff, dt=dt, overlaps=tab)
    st, L = _uniform_gas(T, col.K, N, s0, KB * 8e-6, rng, n_init=0)
    _collide(col, st, T, L, rng, 1500, s0)
    assert col.stats["released"] < 0.01 * N
    b = st.status == Status.BOUND
    Tp = M_RB * (st.vx[b].var() + st.vy[b].var()) / (2 * KB)
    nb = int(T.tab.n_bound(np.array([s0]), T.tilt)[0])
    E = T.ER * T.tab.energy(np.full(nb, s0), np.arange(nb))
    eps_ref = E - E[0]
    nb_ = int(b.sum())
    eps = T.ER * (T.tab.energy(np.full(nb_, s0), st.n[b]) - T.tab.energy(np.full(nb_, s0), np.zeros(nb_, int)))
    Tpar = thermal_band_temperature(eps_ref, eps.mean())
    assert abs(Tp / Tpar - 1) < 0.05, (Tp * 1e6, Tpar * 1e6)
    assert 3e-6 < Tp < 8e-6                               # energy moved from radial to axial
