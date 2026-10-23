"""Level-0 tests: two-beam lattice field."""
import numpy as np

from rbdsmc.config import load_config
from rbdsmc.constants import KB, UK, M_RB, G_ACC
from rbdsmc.physics.lattice_field import TwoBeamLattice

LAM, W0 = 1.064e-6, 20e-6
K = 2 * np.pi / LAM
ZR = np.pi * W0 ** 2 / LAM
RNG = np.random.default_rng(1)


def _lattice(*overrides):
    return TwoBeamLattice.from_config(load_config(None, overrides=list(overrides)))


def test_balanced_on_axis_depth_and_offset():
    lat = _lattice()
    z = np.linspace(4e-3, 9e-3, 11)
    L = lat.local(np.column_stack([0 * z, 0 * z, z]))
    assert np.allclose(L.U1, L.U2, rtol=1e-12)
    assert np.allclose(L.Voff, 0.0, atol=1e-12 * L.U1.max())
    assert np.allclose(L.Ulat, 4 * L.U1, rtol=1e-12)


def test_depths_match_framing_table():
    """U_lat on axis (uK) for the framing-table points, plus the simplest case at 7 mm."""
    table = {(0.3, 5): 15.6, (0.3, 8): 6.3, (0.8, 5): 41.6, (0.8, 8): 16.8, (0.5, 7): 13.6}
    for (P, zmm), ref in table.items():
        lat = _lattice(f"optics.beam_up.power_W={P}", f"optics.beam_down.power_W={P}")
        U = lat.local(np.array([[0, 0, zmm * 1e-3]])).Ulat[0] / UK
        assert abs(U / ref - 1) < 0.01, (P, zmm, U, ref)


def test_potential_equals_coherent_sum():
    lat = _lattice("optics.theta_rad=5e-3", "optics.phase_offset_rad=0.7")
    r = np.column_stack([RNG.normal(0, 60e-6, 500), RNG.normal(0, 60e-6, 500),
                         RNG.uniform(6e-3, 8e-3, 500)])
    V1, V2 = lat.potential(r), lat.potential_direct(r)
    assert np.allclose(V1, V2, rtol=1e-9, atol=1e-9 * np.abs(V1).max())


def test_phase_matches_analytic_on_common_axis():
    lat = _lattice()
    x, y = RNG.normal(0, 80e-6, 200), RNG.normal(0, 80e-6, 200)
    z = RNG.uniform(5e-3, 9e-3, 200)
    Phi = lat.local(np.column_stack([x, y, z])).Phi
    R = z * (1 + (ZR / z) ** 2)
    expected = 2 * K * z + K * (x ** 2 + y ** 2) / R - 2 * np.arctan(z / ZR)
    assert np.max(np.abs(Phi - expected)) < 1e-6


def test_site_spacing_on_axis():
    lat = _lattice()
    j0 = lat.site_index_near(7e-3)
    z, _ = lat.site_surface(np.zeros(3), np.zeros(3), np.array([j0, j0 + 1, j0 + 1000]))
    d1 = z[1] - z[0]
    # Phi = 2kz - 2 zeta  ->  local period pi / (k - dzeta/dz)
    dzeta = ZR / (ZR ** 2 + z[0] ** 2)
    assert abs(d1 / (np.pi / (K - dzeta)) - 1) < 1e-7
    assert abs(d1 / (LAM / 2) - 1) < 1e-4
    assert abs((z[2] - z[0]) / 1000 / (LAM / 2) - 1) < 1e-4


def test_site_surface_is_on_antinode():
    lat = _lattice("optics.theta_rad=2e-3")
    j0 = lat.site_index_near(7e-3)
    x, y = RNG.normal(0, 80e-6, 100), RNG.normal(0, 80e-6, 100)
    z, L = lat.site_surface(x, y, j0)
    assert np.max(np.abs(L.Phi - 2 * np.pi * j0)) < 1e-8
    assert np.all(lat.site_index(np.column_stack([x, y, z])) == j0)
    # V at the surface is a local minimum along z: V(z +- d/4) > V(z).  Only meaningful
    # where the lattice modulation exceeds the gravity energy over d/4 (inside the beam).
    inside = x ** 2 + y ** 2 < (120e-6) ** 2
    r = np.column_stack([x, y, z])[inside]
    for dz in (+LAM / 8, -LAM / 8):
        assert np.all(lat.potential(r + [0, 0, dz]) > lat.potential(r))


def test_site_surface_curvature():
    """Planes bend as z_j(rho) - z_j(0) = -rho^2 / (2 R)  (dome, radius ~ z)."""
    lat = _lattice()
    j0 = lat.site_index_near(7e-3)
    rho = np.array([0.0, 20e-6, 50e-6, 100e-6])
    z, _ = lat.site_surface(rho, 0 * rho, j0)
    R = z[0] * (1 + (ZR / z[0]) ** 2)
    sag = z - z[0]
    assert np.allclose(sag[1:], -rho[1:] ** 2 / (2 * R), rtol=2e-3)
    assert sag[-1] < -0.5e-6          # more than a lattice spacing across the beam


def test_tilted_beams_bisector_and_plane_tilt():
    th, z0 = 1e-2, 7e-3
    lat = _lattice(f"optics.theta_rad={th}")
    x = np.linspace(-20e-6, 90e-6, 22001)
    L = lat.local(np.column_stack([x, 0 * x, np.full_like(x, z0)]))
    x_b = x[np.argmax(L.Ulat)]
    assert abs(x_b - z0 * np.tan(th) / 2) < 0.1e-6            # lattice centre on bisector
    i = np.argmin(np.abs(x - x_b))
    assert abs(L.Voff[i]) < 1e-3 * L.Ulat[i]                     # balanced on the bisector
    w = W0 * np.sqrt(1 + (z0 / ZR) ** 2)
    L0 = _lattice().local(np.array([[0, 0, z0]]))
    assert abs(L.Ulat[i] / L0.Ulat[0] / np.exp(-(th * z0) ** 2 / (2 * w ** 2)) - 1) < 2e-3
    # plane slope on the bisector: dz/dx = -tan(theta/2) (curvature slopes cancel there)
    j0 = lat.site_index_near(z0)
    xs = x_b + np.array([-1e-6, 1e-6])
    zs, _ = lat.site_surface(xs, 0 * xs, j0)
    slope = (zs[1] - zs[0]) / (xs[1] - xs[0])
    assert abs(slope / (-np.tan(th / 2)) - 1) < 2e-2


def test_gradients_vs_finite_differences():
    lat = _lattice("optics.theta_rad=3e-3")
    r = np.column_stack([RNG.normal(10e-6, 60e-6, 60), RNG.normal(0, 60e-6, 60),
                         RNG.uniform(6e-3, 8e-3, 60)])
    L = lat.local(r, grad=True)
    h = 1e-9
    for k in range(3):
        dr = np.zeros(3); dr[k] = h
        Lp, Lm = lat.local(r + dr), lat.local(r - dr)
        for name in ("U1", "U2", "Ulat", "Voff", "Phi"):
            fd = (getattr(Lp, name) - getattr(Lm, name)) / (2 * h)
            an = {"U1": L.gU1, "U2": L.gU2, "Ulat": L.gUlat, "Voff": L.gVoff, "Phi": L.gPhi}[name][:, k]
            scale = np.abs(an).max() + 1e-300
            assert np.allclose(an, fd, rtol=1e-4, atol=1e-5 * scale), (name, k)


def test_averaged_force_and_gravity():
    lat = _lattice()
    r = np.column_stack([RNG.normal(0, 60e-6, 30), RNG.normal(0, 60e-6, 30),
                         RNG.uniform(1e-3, 8e-3, 30)])
    F = lat.force_avg(r)
    h = 1e-9
    for k in range(3):
        dr = np.zeros(3); dr[k] = h
        fd = -(lat.potential_avg(r + dr) - lat.potential_avg(r - dr)) / (2 * h)
        assert np.allclose(F[:, k], fd, rtol=1e-5, atol=1e-6 * np.abs(F).max())
    # far from the beam only gravity acts
    Ff = lat.force_avg(np.array([[5e-3, 0, 7e-3]]))
    assert np.allclose(Ff[0], [0, 0, -M_RB * G_ACC], rtol=1e-9, atol=1e-40)
