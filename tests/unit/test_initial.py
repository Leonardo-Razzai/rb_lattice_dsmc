"""Level-1 tests: initial-state sampling against direct quadrature (theta = 0, one site)."""
import numpy as np
from scipy.stats import norm

from rbdsmc.config import load_config
from rbdsmc.constants import KB, M_RB
from rbdsmc.initial import sample_initial, sample_sites, total_energy
from rbdsmc.physics.escape import escape_energies, EscapeTable
from rbdsmc.physics.potentials import SemiclassicalTrap
from rbdsmc.state import Status

_C = {}


def setup():
    if not _C:
        cfg = load_config(None)
        T = SemiclassicalTrap(cfg)
        _C.update(cfg=cfg, T=T)
    return _C["cfg"], _C["T"]


def _quadrature(T, j, kT, n_bands):
    """Z_n, <rho^2>_n and bound fraction by radial quadrature (cylindrical symmetry)."""
    z0 = float(T.site_z_axis(np.array([j]))[0])
    rho = np.linspace(0, float(T.rho_region(z0)), 4001)
    Z, R2, Fb = [], [], []
    for n in range(n_bands):
        e = T.bound_eval(rho, 0 * rho, j, np.full(len(rho), n), force=False)
        wgt = 2 * np.pi * rho * np.exp(-(e.V - T.E_ref) / kT)
        Z.append(np.trapezoid(wgt, rho))
        R2.append(np.trapezoid(wgt * rho ** 2, rho))
        Fb.append(np.trapezoid(wgt * e.bound, rho))
    return np.array(Z), np.array(R2), np.array(Fb)


def test_single_site_distribution_matches_quadrature():
    cfg, T = setup()
    kT = cfg.cloud.kT0
    N = 20000
    st, info = sample_initial(T, cfg, np.random.default_rng(5), n_test=N,
                              sites=np.full(N, T.j_c), distribution="boltzmann")
    assert info["violations"] == 0
    nb = info["n_cut_max"]
    Z, R2, Fb = _quadrature(T, T.j_c, kT, nb)
    p = Z / Z.sum()
    counts = np.bincount(st.n, minlength=nb)[:nb]
    # chi-square over bands with expected count >= 5
    sel = p * N >= 5
    chi2 = np.sum((counts[sel] - N * p[sel]) ** 2 / (N * p[sel]))
    dof = sel.sum() - 1
    assert chi2 < dof + 5 * np.sqrt(2 * dof), (chi2, dof)
    # mean rho^2 and bound fraction
    rho2 = (st.x - T.x_center(st.z)) ** 2 + st.y ** 2
    rho2_q = R2.sum() / Z.sum()
    assert abs(rho2.mean() / rho2_q - 1) < 4 * rho2.std() / np.sqrt(N) / rho2_q
    fb_q = Fb.sum() / Z.sum()
    fb = np.mean(st.status == Status.BOUND)
    assert abs(fb - fb_q) < 4 * np.sqrt(fb_q * (1 - fb_q) / N), (fb, fb_q)
    # Maxwellian transverse velocities
    for v in (st.vx, st.vy):
        assert abs(v.var() * M_RB / kT - 1) < 4 * np.sqrt(2 / N)


def test_envelope_truncated_gaussian():
    cfg, T = setup()
    rng = np.random.default_rng(7)
    j = sample_sites(T, rng, 20000, cfg.cloud.z_center, cfg.cloud.sigma_z, cfg.region.z_half)
    z = T.site_z_axis(np.unique(j))
    zj = dict(zip(np.unique(j).tolist(), z.tolist()))
    z = np.array([zj[v] for v in j.tolist()])
    a = cfg.region.z_half / cfg.cloud.sigma_z
    std_th = cfg.cloud.sigma_z * np.sqrt(1 - 2 * a * norm.pdf(a) / (2 * norm.cdf(a) - 1))
    assert abs(z.mean() - cfg.cloud.z_center) < 4 * std_th / np.sqrt(len(z))
    assert abs(z.std() / std_th - 1) < 0.02
    assert np.all(np.abs(z - cfg.cloud.z_center) <= cfg.region.z_half + 1e-6)


def test_truncated_distribution_below_escape_energy():
    cfg, T = setup()
    n_max = 8
    esc = EscapeTable(T, T.j_c - 10, T.j_c + 10, n_max, slab_sites=10)
    N = 1500
    st, info = sample_initial(T, cfg, np.random.default_rng(9), n_test=N,
                              sites=np.full(N, T.j_c), distribution="truncated", escape=esc)
    assert np.all(st.status == Status.BOUND)
    E = total_energy(T, st)
    assert np.all(E < esc(st.j, st.n))
    # the truncated sample is the Boltzmann sample conditioned on E < E_esc: the energy
    # distribution of each band follows exp(-E/kT) times the density of states, so the
    # mean energy is below the escape energy by more than ~kT/2
    E_esc0, _, _ = escape_energies(T, T.j_c, 1)
    assert np.mean(E) < E_esc0[0] - 0.5 * cfg.cloud.kT0
