"""Level-1 tests: integrators, release, region exits and fiber arrival."""
import numpy as np

from rbdsmc.config import load_config
from rbdsmc.constants import G_ACC, KB, M_RB, UK
from rbdsmc.dynamics.integrators import Propagator, verlet
from rbdsmc.initial import total_energy
from rbdsmc.physics.potentials import SemiclassicalTrap
from rbdsmc.state import ParticleState, Status

_T = {}


def trap():
    if "t" not in _T:
        _T["t"] = SemiclassicalTrap(load_config(None))
    return _T["t"]


def test_verlet_harmonic_energy_order_and_no_drift():
    """Energy error of velocity Verlet is O(dt^2) and bounded (recorded every step)."""
    w = 2 * np.pi * 100.0
    acc = lambda x: -w * w * x                       # noqa: E731
    errs = []
    for dt in (1e-4, 5e-5):
        n = int(round(1.0 / dt))                     # 1 s = 100 periods
        x, v = np.array([1.0]), np.array([0.0])
        a = acc(x)
        E = np.empty(n)
        for i in range(n):
            x, v, a = verlet(x, v, a, acc, dt, 1)
            E[i] = (0.5 * v[0] ** 2 + 0.5 * w * w * x[0] ** 2) / (0.5 * w * w)
        errs.append(np.max(np.abs(E - 1)))
        q = n // 4
        assert np.max(np.abs(E[-q:] - 1)) < 1.05 * np.max(np.abs(E[:q] - 1))   # no growth
    assert 3.5 < errs[0] / errs[1] < 4.5               # O(dt^2)


def _single(status, x, y, z, vx=0.0, vy=0.0, vz=0.0, j=0, n=0):
    st = ParticleState.empty(1)
    st.status[0] = status
    st.x[0], st.y[0], st.z[0] = x, y, z
    st.vx[0], st.vy[0], st.vz[0] = vx, vy, vz
    st.j[0], st.n[0] = j, n
    return st


def _run_until(prop, st, cond, t_max, t=0.0):
    while t < t_max and not cond(st):
        prop.step(st, t)
        t += prop.dt
    return t


def test_free_fall_far_from_beams_hits_facet():
    T = trap()
    z0 = 7e-3
    st = _single(Status.IN_TRANSIT, 3e-3, 0.0, z0)
    prop = Propagator(T, 25e-6, 10, np.random.default_rng(0))
    prop.init_accelerations(st)
    _run_until(prop, st, lambda s: s.status[0] != Status.IN_TRANSIT, 0.1)
    assert st.status[0] == Status.LOST_FACET
    assert abs(st.t_event[0] - np.sqrt(2 * z0 / G_ACC)) < 5e-6
    assert abs(st.x[0] - 3e-3) < 1e-9


def test_atom_dropped_on_axis_enters_fiber_radially_bound():
    T = trap()
    st = _single(Status.FREE, T.x_center(5e-3), 0.0, 5e-3)
    prop = Propagator(T, 25e-6, 10, np.random.default_rng(0))
    prop.init_accelerations(st)
    E0 = T.free_energy(st.r(), st.v())[0]
    # energy conservation during the fall (check before the facet)
    t = _run_until(prop, st, lambda s: s.z[0] < 0.5e-3, 0.1)
    E1 = T.free_energy(st.r(), st.v())[0]
    assert abs(E1 - E0) < 1e-4 * abs(T.mg * 5e-3)
    _run_until(prop, st, lambda s: s.status[0] not in (Status.FREE, Status.IN_TRANSIT), 0.1, t)
    assert st.status[0] == Status.IN_FIBER and st.fiber_bound[0]
    assert st.entry_rho[0] < 1e-9
    # faster than free fall (attracted downward by the deepening beam) but close to it
    t_ff = np.sqrt(2 * 5e-3 / G_ACC)
    assert 0.9 * t_ff < st.t_event[0] < t_ff


def test_bound_atoms_conserve_energy():
    T = trap()
    rng = np.random.default_rng(3)
    N = 300
    st = ParticleState.empty(N)
    st.status[:] = Status.BOUND
    st.j[:] = T.j_c
    st.n[:] = rng.integers(0, 4, N)
    st.x[:] = T.x_center(T.z_c) + rng.normal(0, 25e-6, N)
    st.y[:] = rng.normal(0, 25e-6, N)
    st.z[:] = T.z_site_c
    st.vx[:] = rng.normal(0, 0.01, N)
    st.vy[:] = rng.normal(0, 0.01, N)
    prop = Propagator(T, 25e-6, 10, rng)
    prop.init_accelerations(st)
    E0 = total_energy(T, st)
    t = 0.0
    for _ in range(800):                              # 20 ms ~ 2 radial periods
        prop.step(st, t)
        t += prop.dt
    still = st.status == Status.BOUND
    assert still.sum() > 0.9 * N
    dE = (total_energy(T, st)[still] - E0[still]) / (KB * 4e-6)
    assert np.max(np.abs(dE)) < 1e-3                  # < 1e-3 kT at 4 uK


def test_release_at_band_edge_conserves_energy_and_falls():
    T = trap()
    # band 6 (top bound band) launched outward: released near ~40 um, then falls
    st = _single(Status.BOUND, T.x_center(T.z_c), 0.0, T.z_site_c, vx=0.02, j=T.j_c, n=6)
    rng = np.random.default_rng(1)
    prop = Propagator(T, 25e-6, 10, rng)
    prop.init_accelerations(st)
    E0 = total_energy(T, st)[0]
    t = 0.0
    while st.status[0] == Status.BOUND and t < 0.05:
        prop.step(st, t)
        t += prop.dt
    assert st.status[0] == Status.FREE and st.released_band[0] == 6
    E1 = total_energy(T, st)[0]
    assert abs(E1 - E0) < 2e-3 * KB * 4e-6
    _run_until(prop, st, lambda s: s.status[0] != Status.FREE, 0.05)
    assert st.status[0] == Status.IN_TRANSIT or st.status[0] == Status.LOST_RADIAL
