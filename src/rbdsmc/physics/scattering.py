"""Band-resolved two-body s-wave collisions in a lattice site (golden rule, quasi-2D).

For two atoms in the same site with bands (n1, n2) and relative radial energy E_rel, the
rate into final bands (f1, f2) is (docs/theory, Eq. golden)

    Gamma_{n1 n2 -> f1 f2} = (S g_c^2 mu / (hbar^3 A)) * k^2 |I_{n1 n2 f1 f2}|^2 * Theta(open),

    g_c = 4 pi hbar^2 a / m,   mu = m/2,   S = 2 (identical bosons),
    I_abcd = int w_a w_b w_c w_d dx   (x = k xi, Wannier functions normalised in x),
    open: E_rel + E_n1 + E_n2 >= E_f1 + E_f2.

In 2D the density of states is energy independent, so the rate depends on the radial
relative speed only through which channels are open.  ``PAIR_RATE_CONSTANT`` is
C = S g_c^2 mu k^2 / hbar^3 (m^2/s): a pair in area A collides at C * W / A with
W = sum_open |I|^2.

``OverlapTable`` caches I_abcd(s) on a log grid; ``harmonic_overlaps`` gives the
harmonic-oscillator limit (used in tests and for the 3D-limit check).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_hermite, gammaln

from ..constants import A_SCAT, HBAR, M_RB
from .bands import overlap_tensor

S_BOSE = 2.0


def coupling_constant(a=A_SCAT, m=M_RB):
    return 4.0 * np.pi * HBAR ** 2 * a / m


def pair_rate_constant(wavelength, a=A_SCAT, m=M_RB):
    """C = S g_c^2 mu k^2 / hbar^3  (m^2/s)."""
    k = 2.0 * np.pi / wavelength
    return S_BOSE * coupling_constant(a, m) ** 2 * (m / 2.0) * k ** 2 / HBAR ** 3


# ----------------------------------------------------------------------------
# Harmonic-oscillator limit
# ----------------------------------------------------------------------------
def harmonic_overlaps(n_max, l=1.0, n_quad=200):
    """I_abcd = int phi_a phi_b phi_c phi_d dx for HO eigenfunctions of length l."""
    xg, wg = hermgauss(n_quad)               # int e^{-x^2} f(x) dx
    # phi_n(x) = (2^n n! sqrt(pi) l)^(-1/2) H_n(x/l) e^{-x^2/2l^2}; four factors give
    # e^{-2 x^2/l^2}: substitute u = sqrt(2) x / l
    u = xg
    x_over_l = u / np.sqrt(2.0)
    H = np.array([eval_hermite(n, x_over_l) for n in range(n_max)])
    lognorm = -0.5 * (np.arange(n_max) * np.log(2.0) + gammaln(np.arange(n_max) + 1)
                      + 0.5 * np.log(np.pi) + np.log(l))
    P = H * np.exp(lognorm)[:, None]          # phi_n without the Gaussian factor
    dx_du = l / np.sqrt(2.0)
    return np.einsum("ax,bx,cx,dx,x->abcd", P, P, P, P, wg, optimize=True) * dx_du


def thermal_rate_quasi2d(E_levels, I, kT, n2d, C):
    """Thermal-average collision rate per atom for a 2D gas of density n2d (m^-2) whose
    axial levels are E_levels (J) with overlaps I (x units, so C carries k^2).
    Radial relative energies follow the 2D Boltzmann law p(E) = e^{-E/kT}/kT."""
    E = np.asarray(E_levels)
    p = np.exp(-(E - E[0]) / kT)
    p /= p.sum()
    nb = len(E)
    rate = 0.0
    for n1 in range(nb):
        for n2 in range(nb):
            Ein = E[n1] + E[n2]
            # P(open) for channel f with threshold dE = E_f - Ein:  exp(-max(dE,0)/kT)
            Ef = E[:, None] + E[None, :]
            popen = np.exp(-np.maximum(Ef - Ein, 0.0) / kT)
            W = np.sum(I[n1, n2] ** 2 * popen)
            rate += p[n1] * p[n2] * W
    return n2d * C * rate


# ----------------------------------------------------------------------------
# Lattice overlap table
# ----------------------------------------------------------------------------
class OverlapTable:
    """I_abcd(s) for a, b, c, d < n_max on a log-spaced s grid; linear interpolation in ln s.
    Values in x units (multiply by k for 1/m)."""

    def __init__(self, s_grid, I):
        self.s_grid = np.asarray(s_grid, float)
        self.u = np.log(self.s_grid)
        self.I = np.asarray(I, float)                     # (n_s, n, n, n, n)
        self.I2 = self.I ** 2
        self.n_max = self.I.shape[1]
        self.W_full = self.I2.sum(axis=(3, 4))           # (n_s, n, n): all final channels

    @classmethod
    def compute(cls, n_max=16, s_min=1.0, s_max=1000.0, n_s=48, n_q=32, verbose=False):
        s_grid = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_s))
        I = np.empty((n_s, n_max, n_max, n_max, n_max))
        for i, s in enumerate(s_grid):
            I[i] = overlap_tensor(s, n_max, n_q=n_q)
            if verbose:
                print(f"  overlaps s = {s:8.2f} ({i + 1}/{n_s})", flush=True)
        return cls(s_grid, I)

    @classmethod
    def load_or_compute(cls, cache_dir="results/cache", verbose=False, **params):
        p = dict(n_max=16, s_min=1.0, s_max=1000.0, n_s=48, n_q=32)
        p.update(params)
        key = hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()[:10]
        path = Path(cache_dir) / f"overlaps_{key}.npz"
        if path.exists():
            z = np.load(path)
            return cls(z["s_grid"], z["I"])
        tab = cls.compute(verbose=verbose, **p)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, s_grid=tab.s_grid, I=tab.I)
        return tab

    def _weights(self, s):
        u = np.log(np.clip(s, self.s_grid[0], self.s_grid[-1]))
        f = (u - self.u[0]) / (self.u[1] - self.u[0])
        i0 = np.clip(np.floor(f).astype(np.int64), 0, len(self.u) - 2)
        return i0, (f - i0)

    def I2_pairs(self, s, n1, n2):
        """|I_{n1 n2 f1 f2}|^2 for all (f1, f2): shape (P, n_max, n_max)."""
        i0, t = self._weights(np.asarray(s, float))
        a = self.I2[i0, n1, n2]
        b = self.I2[i0 + 1, n1, n2]
        return (1 - t)[:, None, None] * a + t[:, None, None] * b

    def W_full_at(self, s, n1, n2):
        i0, t = self._weights(np.asarray(s, float))
        return (1 - t) * self.W_full[i0, n1, n2] + t * self.W_full[i0 + 1, n1, n2]

    def majorant_by_band(self, s_hi):
        """M[n] = max over m1, m2 <= n and all s <= s_hi of W_full (monotone in n)."""
        k = np.searchsorted(self.s_grid, s_hi, side="right") + 1
        Wmax_s = self.W_full[:max(k, 2)].max(axis=0)           # (n, n)
        M = np.empty(self.n_max)
        for n in range(self.n_max):
            M[n] = Wmax_s[:n + 1, :n + 1].max()
        return M
