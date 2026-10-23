"""Level-0 tests: semiclassical trap potentials."""
import numpy as np

from rbdsmc.config import load_config
from rbdsmc.constants import M_RB, G_ACC, UK
from rbdsmc.physics.potentials import SemiclassicalTrap

_TRAPS = {}


def trap(theta=0.0):
    if theta not in _TRAPS:
        cfg = load_config(None, overrides=[f"optics.theta_rad={theta}"])
        _TRAPS[theta] = SemiclassicalTrap(cfg)
    return _TRAPS[theta]


def test_bound_force_is_minus_gradient_along_surface():
    for th in (0.0, 1e-2):
        T = trap(th)
        rng = np.random.default_rng(0)
        N = 40
        x = T.x_center(T.z_c) + rng.normal(0, 50e-6, N)
        y = rng.normal(0, 50e-6, N)
        n = rng.integers(0, 5, N)
        e = T.bound_eval(x, y, T.j_c, n)
        h = 1e-8
        for k, (dx, dy) in enumerate([(h, 0), (0, h)]):
            Vp = T.bound_eval(x + dx, y + dy, T.j_c, n, force=False).V
            Vm = T.bound_eval(x - dx, y - dy, T.j_c, n, force=False).V
            fd = -(Vp - Vm) / (2 * h)
            scale = np.abs(e.F).max()
            assert np.allclose(e.F[:, k], fd, rtol=1e-4, atol=1e-4 * scale), (th, k)


def test_band0_radial_frequency_matches_analytic():
    """omega^2 = omega_r^2 <cos^2>_0 - g/R  near the lattice centre."""
    T = trap(0.0)
    zc = T.z_site_c
    x = np.array([-2e-6, 0.0, 2e-6])
    e = T.bound_eval(x, 0 * x, T.j_c, np.zeros(3, int), force=False)
    k_num = (e.V[0] - 2 * e.V[1] + e.V[2]) / (2e-6) ** 2
    w = T.beam_radius(zc)
    R = zc * (1 + (T.lat.up.zR / zc) ** 2)
    cos2 = -T.tab.dEds(e.s[1:2], np.array([0]))[0]
    k_ana = 4 * e.Ulat[1] / w ** 2 * cos2 - M_RB * G_ACC / R
    assert abs(k_num / k_ana - 1) < 2e-3


def test_band_dependent_confinement_ordering():
    """Higher bands feel weaker radial confinement (<cos^2>_n decreases with n)."""
    T = trap(0.0)
    x = np.array([0.0, 30e-6])
    ks = []
    for n in range(6):
        e = T.bound_eval(x, 0 * x, T.j_c, np.full(2, n), force=False)
        ks.append(e.V[1] - e.V[0])
    assert all(ks[i] > ks[i + 1] for i in range(5))


def test_release_threshold_and_energy_conservation():
    T = trap(0.0)
    x = np.linspace(0, 300e-6, 3001)
    n_top = 6
    e = T.bound_eval(x, 0 * x, T.j_c, np.full(len(x), n_top), force=False)
    i = np.argmax(~e.bound)                       # first point where band 6 is unbound
    assert 0 < i < len(x) - 1 and e.bound[:i].all()
    _, emax = T.tab.edges(e.s[i - 1:i + 1], np.full(2, n_top))
    assert emax[0] < -T.tilt <= emax[1]
    # conversion bound -> free conserves energy exactly
    vx, vy = 3e-3, -1e-3
    E_b = 0.5 * M_RB * (vx * vx + vy * vy) + e.V[i]
    vz = np.sqrt(T.release_vz2(e.E_band[i:i + 1], e.Ulat[i:i + 1]))
    r = np.array([[x[i], 0.0, e.z[i]]])
    E_f = T.free_energy(r, np.array([[vx, vy, vz[0]]]))[0]
    assert abs(E_f - E_b) < 1e-9 * abs(E_b)


def test_all_bound_bands_released_at_beam_edge():
    T = trap(0.0)
    w = T.beam_radius(T.z_c)
    x = np.array([3 * w])
    for n in range(8):
        e = T.bound_eval(x, 0 * x, T.j_c, np.array([n]), force=False)
        assert not e.bound[0]


def test_energy_reference_and_fiber_test():
    T = trap(0.0)
    e = T.bound_eval(np.array([0.0]), np.array([0.0]), T.j_c, np.array([0]), force=False)
    # band-0 energy above the classical minimum = zero-point energy minus anharmonic shift
    E0 = (e.V[0] - T.E_ref) / UK
    hw = 2 * np.sqrt(e.s[0]) * T.ER / UK
    assert 0.4 * hw < E0 < 0.5 * hw
    ok, _ = T.fiber_radially_bound(np.array([0.0, 10e-6, 40e-6]), np.zeros(3))
    assert ok.tolist() == [True, True, False]
    # transverse speed above the escape speed is not bound
    v_esc = np.sqrt(2 * T.U_fiber / M_RB)
    ok, _ = T.fiber_radially_bound(np.array([0.0]), np.array([(1.01 * v_esc) ** 2]))
    assert not ok[0]
