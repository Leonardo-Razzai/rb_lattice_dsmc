"""Level-0 tests: lattice band structure."""
import numpy as np
from scipy.special import mathieu_a, mathieu_b

from rbdsmc.physics.bands import (BandTable, band_data, bloch_states, harmonic_estimate,
                                  overlap_tensor, wannier_functions)


def test_band_edges_match_mathieu():
    """Band n spans [a_n(q), b_{n+1}(q)] - s/2 with q = s/4 (Mathieu characteristic values)."""
    for s in [5.0, 50.0, 160.0, 400.0]:
        d = band_data(s, n_bands=8)
        for n in range(8):
            lo = mathieu_a(n, s / 4) - s / 2
            hi = mathieu_b(n + 1, s / 4) - s / 2
            assert abs(d["E_min"][n] - lo) < 1e-6 * max(1, abs(lo)), (s, n)
            assert abs(d["E_max"][n] - hi) < 1e-6 * max(1, abs(hi)), (s, n)


def test_independent_finite_difference_check():
    """Edges at q = 0 from a real-space finite-difference Hamiltonian on one period."""
    s, N = 100.0, 800
    x = np.linspace(0, np.pi, N, endpoint=False)
    h = x[1] - x[0]
    H = (np.diag(np.full(N, 2.0)) - np.diag(np.ones(N - 1), 1) - np.diag(np.ones(N - 1), -1))
    H[0, -1] = H[-1, 0] = -1.0          # periodic -> q = 0
    H = H / h ** 2 + np.diag(-s * np.cos(x) ** 2)
    E_fd = np.linalg.eigvalsh(H)[:5]
    E_pw, _ = bloch_states(s, 0.0, 5)
    # second-order FD error ~ h^2: observed < 2.2e-3 E_R for the lowest five bands
    assert np.allclose(E_fd, E_pw, rtol=0, atol=5e-3)


def test_deep_lattice_harmonic_limit():
    s = 1000.0
    d = band_data(s, n_bands=6)
    hw = 2 * np.sqrt(s)
    for n in range(4):
        # next correction is O(n^3 / sqrt(s)) E_R; require agreement to 1% of hbar*omega
        assert abs(d["E_mean"][n] - harmonic_estimate(s, n)) < 0.01 * hw, n
    # ground band essentially flat at large depth
    assert d["E_max"][0] - d["E_min"][0] < 1e-8


def test_hellmann_feynman_derivative():
    for s in [30.0, 200.0, 800.0]:
        h = 1e-3 * s
        d0 = band_data(s, n_bands=12)
        dp = band_data(s + h, n_bands=12)
        dm = band_data(s - h, n_bands=12)
        fd = (dp["E_mean"] - dm["E_mean"]) / (2 * h)
        assert np.allclose(d0["dEds"], fd, rtol=1e-5, atol=1e-6)
        # -<cos^2> lies in (-1, 0) and increases with band index for bound bands
        assert np.all((d0["dEds"] < 0) & (d0["dEds"] > -1))


def test_bound_band_counts_match_framing():
    """Bands with bottom below the barrier top: 8, 5, 13, 9 for s = 160, 65, 428, 173."""
    for s, nb in [(160.0, 8), (65.0, 5), (428.0, 13), (173.0, 9)]:
        d = band_data(s, n_bands=20)
        assert int(np.sum(d["E_min"] < 0)) == nb, s


def test_wannier_normalised_real_parity_localised():
    s = 140.0
    x, W = wannier_functions(s, 6)
    dx = x[1] - x[0]
    assert np.allclose(np.sum(W ** 2, axis=1) * dx, 1.0)
    for n in range(6):
        assert np.allclose(W[n, ::-1], (-1) ** n * W[n], atol=1e-8), n
        far = np.abs(x) > 1.5 * np.pi            # beyond the neighbouring sites
        assert np.sum(W[n, far] ** 2) * dx < 1e-4, n


def test_overlap_integrals():
    s = 1000.0
    I = overlap_tensor(s, 4)
    # ground band: 1/(sqrt(2 pi) l), l = s^(-1/4) in x units  (harmonic limit)
    I0 = s ** 0.25 / np.sqrt(2 * np.pi)
    assert abs(I[0, 0, 0, 0] / I0 - 1) < 0.02
    # parity selection rule: zero unless a+b+c+d is even
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for d in range(4):
                    if (a + b + c + d) % 2:
                        assert abs(I[a, b, c, d]) < 1e-10
    # harmonic-oscillator values: I_0011 = I0/2, I_1111 = 3 I0/4
    assert abs(I[0, 0, 1, 1] / (I0 / 2) - 1) < 0.03
    assert abs(I[1, 1, 1, 1] / (3 * I0 / 4) - 1) < 0.04


def test_band_table_interpolation_accuracy():
    tab = BandTable.compute(s_min=10.0, s_max=1000.0, n_s=120, n_bands=12)
    s_test = np.exp(np.random.default_rng(3).uniform(np.log(11), np.log(990), 15))
    for s in s_test:
        d = band_data(s, n_bands=12)
        n = np.arange(12)
        sv = np.full(12, s)
        assert np.allclose(tab.energy(sv, n), d["E_mean"], atol=2e-4 * s ** 0.5)
        assert np.allclose(tab.dEds(sv, n), d["dEds"], atol=2e-4)
        # spline derivative of E agrees with the tabulated Hellmann-Feynman values
        assert np.allclose(tab._eval("E_mean", sv, n, deriv=1), d["dEds"], atol=2e-3)
    # n_bound with tilt: at s = 65 the top bound band lies [-0.61, -0.41] uK ~ [-6.3, -4.2] E_R
    assert tab.n_bound(65.0, tilt=0.284)[0] in (4, 5)
