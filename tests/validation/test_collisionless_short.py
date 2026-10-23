"""Level-2 preview: short collisionless evolution of a sampled cloud (simplest case).

Checks (exact properties of the collisionless semiclassical dynamics):
  * every propagated atom conserves its energy (to Verlet accuracy),
  * every atom below its escape energy at t = 0 is still BOUND at the end,
  * the surviving fraction is >= the predicted fraction (angular momentum can only
    keep additional atoms) and only slightly above it.
"""
import numpy as np

try:
    import pytest
    slow = pytest.mark.slow
except ImportError:                       # fallback runner
    slow = lambda f: f                    # noqa: E731

from rbdsmc.config import load_config
from rbdsmc.dynamics.integrators import Propagator
from rbdsmc.initial import sample_initial, total_energy
from rbdsmc.physics.escape import escape_table_for
from rbdsmc.physics.potentials import SemiclassicalTrap
from rbdsmc.state import Status


@slow
def test_collisionless_energy_and_trapping():
    cfg = load_config("configs/test_simplest.yaml")
    T = SemiclassicalTrap(cfg)
    esc = escape_table_for(T, cfg)
    N = 1000
    st, _ = sample_initial(T, cfg, np.random.default_rng(21), n_test=N)
    prop = Propagator(T, cfg.numerics.dt, cfg.numerics.free_substeps, np.random.default_rng(22))
    prop.init_accelerations(st)
    E0 = total_energy(T, st)
    b0 = st.status == Status.BOUND
    trapped0 = b0.copy()
    trapped0[b0] = E0[b0] < esc(st.j[b0], st.n[b0])
    t = 0.0
    for _ in range(1600):                 # 40 ms
        prop.step(st, t)
        t += prop.dt
    # atoms in the trap: |dE| < 1e-3 kT.  Falling atoms gain ~ m g z_c (0.7 mK) and oscillate
    # radially at up to ~2.4 kHz near the facet; with h = 2.5 us Verlet gives relative errors
    # ~ (omega h)^2 / 8 ~ 2e-4 of the radial energy: require |dE| < 1e-4 m g z_c (~70 nK),
    # negligible against the 246 uK guide depth used in the fiber test.
    dE = total_energy(T, st) - E0
    in_trap = np.isin(st.status, [Status.BOUND, Status.FREE])
    transit = st.status == Status.IN_TRANSIT
    assert np.max(np.abs(dE[in_trap])) < 1e-3 * cfg.cloud.kT0
    if transit.any():
        assert np.max(np.abs(dE[transit])) < 1e-4 * T.mg * cfg.cloud.z_center
    assert np.all(st.status[trapped0] == Status.BOUND)
    f_pred = trapped0.mean()
    f_now = np.mean(st.status == Status.BOUND)
    assert f_pred <= f_now <= f_pred + 0.03
    # no atom reaches the facet before the free-fall time from the bottom of the region
    hit = np.isin(st.status, [Status.IN_FIBER, Status.LOST_FACET])
    if hit.any():
        z_low = cfg.cloud.z_center - cfg.region.z_half
        assert np.min(st.t_event[hit]) > 0.8 * np.sqrt(2 * z_low / 9.80665) * 0.9
