"""Level-0 tests: transverse modes and beams."""
import numpy as np

from rbdsmc.physics.modes import (GaussianTEM00, HermiteGauss, LaguerreGauss, ModeSum,
                                  build_mode)
from rbdsmc.physics.beams import Beam, make_beam_pair
from rbdsmc.config import load_config

LAM, W0 = 1.064e-6, 20e-6
ZR = np.pi * W0 ** 2 / LAM
RNG = np.random.default_rng(0)


def _grid(s, half=4.0, n=241):
    w = W0 * np.sqrt(1 + (s / ZR) ** 2)
    x = np.linspace(-half * w, half * w, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X.ravel(), Y.ravel(), (x[1] - x[0]) ** 2


def _overlap(m1, m2, s):
    X, Y, dA = _grid(s)
    S = np.full_like(X, s)
    return np.sum(np.conj(m1.field(X, Y, S)) * m2.field(X, Y, S)) * dA


# --------------------------------------------------------------------------- geometry
def test_gaussian_geometry_formulas():
    m = GaussianTEM00(LAM, W0)
    assert abs(m.zR - 1.1810e-3) < 1e-6
    for s in [-7e-3, 0.0, 5e-3, 8e-3]:
        assert np.isclose(m.w(s), W0 * np.sqrt(1 + (s / ZR) ** 2))
        assert np.isclose(m.gouy(s), np.arctan(s / ZR))
        if s != 0:
            assert np.isclose(1 / m.inv_R(s), s * (1 + (ZR / s) ** 2))
    assert m.inv_R(0.0) == 0.0


def test_normalisation_and_orthogonality():
    modes = [GaussianTEM00(LAM, W0), LaguerreGauss(LAM, W0, 1, 0), LaguerreGauss(LAM, W0, 0, 2),
             LaguerreGauss(LAM, W0, 2, 1), HermiteGauss(LAM, W0, 1, 0), HermiteGauss(LAM, W0, 2, 1)]
    for s in [0.0, 7e-3, -7e-3]:
        for m in modes:
            assert abs(_overlap(m, m, s) - 1.0) < 2e-4, (m.describe(), s)
    lg = modes[:4]
    for i in range(len(lg)):
        for j in range(i + 1, len(lg)):
            assert abs(_overlap(lg[i], lg[j], 7e-3)) < 1e-4


def test_lg00_and_hg00_equal_tem00():
    g = GaussianTEM00(LAM, W0)
    x, y = RNG.normal(0, 100e-6, 200), RNG.normal(0, 100e-6, 200)
    s = RNG.uniform(-8e-3, 8e-3, 200)
    for other in [LaguerreGauss(LAM, W0, 0, 0), HermiteGauss(LAM, W0, 0, 0),
                  ModeSum([LaguerreGauss(LAM, W0, 0, 0)], [2.0 + 0j])]:
        assert np.allclose(other.field(x, y, s), g.field(x, y, s), rtol=1e-12, atol=0)


def test_gouy_phase_order():
    """On axis the phase relative to -k s is (N+1) zeta(s)."""
    for m, N in [(GaussianTEM00(LAM, W0), 0), (LaguerreGauss(LAM, W0, 1, 0), 2),
                 (LaguerreGauss(LAM, W0, 3, 0), 6)]:
        for s in [2e-3, 7e-3]:
            u = m.field(np.array([0.0]), np.array([0.0]), np.array([s]))[0]
            ph = np.angle(u * np.exp(1j * m.k * s))
            # LG_p0 on axis: L_p(0) = 1 > 0, so the sign is +; compare modulo 2 pi
            d = np.angle(np.exp(1j * (ph - (N + 1) * np.arctan(s / ZR))))
            assert abs(d) < 1e-9, (m.describe(), s, d)


def test_phase_decomposition_consistent():
    """field = residual * exp(i ref_phase) and ref_phase matches the analytic TEM00 phase."""
    m = GaussianTEM00(LAM, W0)
    x, y, s = 30e-6, -50e-6, 7e-3
    rho2 = x * x + y * y
    R = s * (1 + (ZR / s) ** 2)
    expected = -m.k * s - m.k * rho2 / (2 * R) + np.arctan(s / ZR)
    assert np.isclose(m.ref_phase(x, y, s), expected, rtol=0, atol=1e-9)
    assert np.isclose(np.angle(m.residual(np.array([x]), np.array([y]), np.array([s]))[0]), 0.0)


def test_gaussian_gradients_vs_finite_differences():
    m = GaussianTEM00(LAM, W0)
    x, y = RNG.normal(0, 80e-6, 50), RNG.normal(0, 80e-6, 50)
    s = RNG.uniform(4e-3, 9e-3, 50)
    ga = m.residual_grad(x, y, s)
    gf = super(GaussianTEM00, m).residual_grad(x, y, s)   # base-class finite differences
    for a, f in zip(ga, gf):
        assert np.allclose(a, f, rtol=1e-5, atol=1e-6 * np.max(np.abs(a)))
    pa = m.ref_phase_grad(x, y, s)
    h = 1e-9
    pf = ((m.ref_phase(x + h, y, s) - m.ref_phase(x - h, y, s)) / (2 * h),
          (m.ref_phase(x, y + h, s) - m.ref_phase(x, y - h, s)) / (2 * h),
          (m.ref_phase(x, y, s + h) - m.ref_phase(x, y, s - h)) / (2 * h))
    for a, f in zip(pa, pf):
        assert np.allclose(a, f, rtol=1e-6, atol=1e-2)


# --------------------------------------------------------------------------- beams
def test_beam_power_integral():
    up, down = make_beam_pair(load_config(None).optics)
    for beam in (up, down):
        for z in [0.0, 5e-3, 7e-3]:
            w = W0 * np.sqrt(1 + (z / ZR) ** 2)
            x = np.linspace(-4 * w, 4 * w, 301)
            X, Y = np.meshgrid(x, x, indexing="ij")
            r = np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)
            P = beam.intensity(r).sum() * (x[1] - x[0]) ** 2
            assert abs(P / beam.power - 1) < 1e-4


def test_peak_intensity_value():
    up, _ = make_beam_pair(load_config(None).optics)
    z = 7e-3
    w = W0 * np.sqrt(1 + (z / ZR) ** 2)
    I0 = up.intensity(np.array([[0.0, 0.0, z]]))[0]
    assert np.isclose(I0, 2 * up.power / (np.pi * w ** 2))


def test_propagation_directions_and_phase_sign():
    """Phase of a beam decreases by ~k per metre along its propagation direction."""
    cfg = load_config(None, overrides=["optics.theta_rad=0.0"])
    up, down = make_beam_pair(cfg.optics)
    z = 7e-3
    dz = 1e-9
    r0, r1 = np.array([[0, 0, z]]), np.array([[0, 0, z + dz]])
    k = 2 * np.pi / LAM
    d_up = (up.field_data(r1).phase_ref - up.field_data(r0).phase_ref)[0] / dz
    d_dn = (down.field_data(r1).phase_ref - down.field_data(r0).phase_ref)[0] / dz
    assert abs(d_up / (-k) - 1) < 1e-3     # beam_up travels +z
    assert abs(d_dn / (+k) - 1) < 1e-3     # beam_down travels -z


def test_tilted_beam_axis_position():
    th = 1e-2
    _, down = make_beam_pair(load_config(None, overrides=[f"optics.theta_rad={th}"]).optics)
    z = 7e-3
    x = np.linspace(0, 150e-6, 30001)
    r = np.stack([x, np.zeros_like(x), np.full_like(x, z)], 1)
    xmax = x[np.argmax(down.intensity(r))]
    # axis passes through origin with direction (sin th, 0, cos th): x = z tan th
    assert abs(xmax - z * np.tan(th)) < 0.02e-6
    assert np.isclose(down.e2 @ np.array([0, 1, 0]), -1.0)     # documented handedness


def test_beam_gradients_lab_frame():
    th = 3e-3
    up, down = make_beam_pair(load_config(None, overrides=[f"optics.theta_rad={th}"]).optics)
    r = np.column_stack([RNG.normal(0, 80e-6, 40), RNG.normal(0, 80e-6, 40),
                         RNG.uniform(6e-3, 8e-3, 40)])
    for b in (up, down):
        fd = b.field_data(r, grad=True)
        h = 1e-9
        for k in range(3):
            dr = np.zeros(3); dr[k] = h
            I_p, I_m = b.intensity(r + dr), b.intensity(r - dr)
            gI = (I_p - I_m) / (2 * h)
            assert np.allclose(fd.grad_intensity[:, k], gI, rtol=1e-4,
                               atol=1e-6 * np.abs(fd.grad_intensity).max())
            ph_p = b.field_data(r + dr).phase_ref
            ph_m = b.field_data(r - dr).phase_ref
            assert np.allclose(fd.grad_phase_ref[:, k], (ph_p - ph_m) / (2 * h), rtol=1e-5, atol=5.0)


def test_build_mode_from_config():
    m = build_mode([{"type": "TEM00"}], LAM, W0)
    assert isinstance(m, GaussianTEM00)
    ms = build_mode([{"type": "LG", "p": 0, "l": 0, "coeff": [3.0, 0.0]},
                     {"type": "LG", "p": 1, "l": 0, "coeff": [0.0, 4.0]}], LAM, W0)
    assert isinstance(ms, ModeSum)
    assert np.allclose(np.abs(ms.coeffs) ** 2, [0.36, 0.64])
    assert abs(_overlap(ms, ms, 7e-3) - 1) < 2e-4
