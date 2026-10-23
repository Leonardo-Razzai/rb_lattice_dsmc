"""Band structure of the 1D lattice  V(x) = -s cos^2(x),  x = k xi,  energies in E_R.

Energies are measured from the lattice maximum (V = 0 where cos = 0), so the bottom of the
lattice is at -s and a band is energetically bound if it lies below 0 (minus the tilt per
half site, applied by the caller).  Everything here is dimensionless and universal: the
same table serves every wavelength and atom once s = U_lat / E_R is known.

Method
------
Bloch states psi_{n,q}(x) = sum_l c_l exp(i (q + 2l) x), q in [-1, 1) (units of k).  In
this basis H is real, symmetric and tridiagonal:
    H_ll = (q + 2l)^2 - s/2,   H_{l,l+1} = -s/4.
Band-averaged quantities (= Wannier-state expectation values) are obtained by averaging
over q with the trapezoid rule on q in [0, 1] (E(q) = E(-q), period 2: spectral accuracy).

Provided
--------
* band_data(s, ...)            : E_mean, E_min, E_max, dE/ds (Hellmann-Feynman) per band
* wannier_functions(s, n_max)  : real, maximally localised Wannier functions on a grid
* overlap_tensor(s, n_max)     : I_abcd = int w_a w_b w_c w_d dx   (x units; times k for SI)
* BandTable                    : cached lookup table vs s with fast vectorised interpolation
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh_tridiagonal


# ----------------------------------------------------------------------------
# Core diagonalisation
# ----------------------------------------------------------------------------
def bloch_states(s: float, q: float, n_bands: int, n_pw: int = 60):
    """Energies (n_bands,) and real coefficient vectors (2 n_pw + 1, n_bands) at quasimomentum q."""
    l = np.arange(-n_pw, n_pw + 1)
    diag = (q + 2.0 * l) ** 2 - 0.5 * s
    off = np.full(len(l) - 1, -0.25 * s)
    E, C = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_bands - 1))
    return E, C


def _lat_potential_expect(C, s):
    """<V> = <-s cos^2 x> for each column of C:  -s/2 - (s/2) sum_l c_l c_{l+1}."""
    return -0.5 * s - 0.5 * s * np.sum(C[:-1] * C[1:], axis=0)


def band_data(s: float, n_bands: int = 40, n_pw: int = 60, n_q: int = 33):
    """Band-averaged energy, edges and Hellmann-Feynman derivative at depth s.

    Returns dict of arrays (n_bands,): E_mean, E_min, E_max, dEds (= <V_lat>/s = -<cos^2>),
    all in units of E_R (dEds dimensionless)."""
    qs = np.linspace(0.0, 1.0, n_q)
    wts = np.full(n_q, 1.0 / (n_q - 1))
    wts[[0, -1]] *= 0.5
    E_all = np.empty((n_q, n_bands))
    V_all = np.empty((n_q, n_bands))
    for i, q in enumerate(qs):
        E, C = bloch_states(s, q, n_bands, n_pw)
        E_all[i] = E
        V_all[i] = _lat_potential_expect(C, s)
    return {
        "E_mean": wts @ E_all,
        "E_min": E_all.min(axis=0),
        "E_max": E_all.max(axis=0),
        "dEds": (wts @ V_all) / s,
    }


def harmonic_estimate(s, n):
    """Deep-lattice expansion  E_n = -s + 2 sqrt(s) (n + 1/2) - (2n^2 + 2n + 1)/4  (E_R)."""
    return -s + 2.0 * np.sqrt(s) * (n + 0.5) - (2.0 * n * n + 2.0 * n + 1.0) / 4.0


# ----------------------------------------------------------------------------
# Wannier functions and overlap integrals
# ----------------------------------------------------------------------------
def wannier_functions(s: float, n_max: int, n_pw: int = 60, n_q: int = 64,
                      half_sites: int = 6, pts_per_site: int = 64):
    """Real Wannier functions w_n(x), n < n_max, centred on the site at x = 0.

    Gauge (Kohn, symmetric 1D potential): Bloch vectors are real; for even bands
    psi_q(0) = sum c_l > 0, for odd bands psi_q'(0) ~ sum c_l (q+2l) > 0.  The q average of
    these Bloch functions is maximally localised; odd bands are multiplied by -i.
    Normalised to int w^2 dx = 1 (x = k xi).  Returns (x, W) with W shape (n_max, len(x))."""
    x = np.linspace(-half_sites * np.pi, half_sites * np.pi, 2 * half_sites * pts_per_site + 1)
    l = np.arange(-n_pw, n_pw + 1)
    qs = -1.0 + (np.arange(n_q) + 0.5) * 2.0 / n_q
    W = np.zeros((n_max, len(x)), dtype=complex)
    for q in qs:
        E, C = bloch_states(s, q, n_max, n_pw)
        kq = q + 2.0 * l
        for n in range(n_max):
            c = C[:, n]
            ref = c.sum() if n % 2 == 0 else (c * kq).sum()
            if ref < 0:
                c = -c
            W[n] += np.exp(1j * np.outer(x, kq)) @ c
    W /= n_q
    W[1::2] *= -1j
    Wr = W.real
    dx = x[1] - x[0]
    Wr /= np.sqrt(np.sum(Wr ** 2, axis=1, keepdims=True) * dx)
    return x, Wr


def overlap_tensor(s: float, n_max: int, **kw):
    """I_abcd = int w_a w_b w_c w_d dx  (x units).  Multiply by k for SI (1/m)."""
    x, W = wannier_functions(s, n_max, **kw)
    dx = x[1] - x[0]
    return np.einsum("ax,bx,cx,dx->abcd", W, W, W, W, optimize=True) * dx


# ----------------------------------------------------------------------------
# Lookup table
# ----------------------------------------------------------------------------
@dataclass
class BandTable:
    """Band data on a log-spaced s grid with fast cubic interpolation in u = ln s.

    Evaluate per atom with ``energy(s, n)``, ``dEds(s, n)``, ``edges(s, n)``; vectorised
    over arrays of s and integer band indices n of the same shape."""
    s_grid: np.ndarray
    E_mean: np.ndarray        # (n_s, n_bands)
    E_min: np.ndarray
    E_max: np.ndarray
    dEds_grid: np.ndarray
    meta: dict

    def __post_init__(self):
        self.u_grid = np.log(self.s_grid)
        self.du = self.u_grid[1] - self.u_grid[0]
        self.n_bands = self.E_mean.shape[1]
        self._coef = {name: CubicSpline(self.u_grid, getattr(self, name), axis=0).c
                      for name in ("E_mean", "E_min", "E_max", "dEds_grid")}

    # -- construction ------------------------------------------------------------
    @classmethod
    def compute(cls, s_min=1.0, s_max=2000.0, n_s=400, n_bands=40, n_pw=60, n_q=33):
        s_grid = np.exp(np.linspace(np.log(s_min), np.log(s_max), n_s))
        out = {k: np.empty((n_s, n_bands)) for k in ("E_mean", "E_min", "E_max", "dEds")}
        for i, s in enumerate(s_grid):
            d = band_data(s, n_bands, n_pw, n_q)
            for k in out:
                out[k][i] = d[k]
        meta = dict(s_min=s_min, s_max=s_max, n_s=n_s, n_bands=n_bands, n_pw=n_pw, n_q=n_q)
        return cls(s_grid, out["E_mean"], out["E_min"], out["E_max"], out["dEds"], meta)

    @classmethod
    def load_or_compute(cls, cache_dir="results/cache", **params):
        defaults = dict(s_min=1.0, s_max=2000.0, n_s=400, n_bands=40, n_pw=60, n_q=33)
        defaults.update(params)
        key = hashlib.sha256(json.dumps(defaults, sort_keys=True).encode()).hexdigest()[:10]
        path = Path(cache_dir) / f"bands_{key}.npz"
        if path.exists():
            z = np.load(path, allow_pickle=False)
            return cls(z["s_grid"], z["E_mean"], z["E_min"], z["E_max"], z["dEds"],
                       json.loads(str(z["meta"])))
        tab = cls.compute(**defaults)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, s_grid=tab.s_grid, E_mean=tab.E_mean, E_min=tab.E_min,
                 E_max=tab.E_max, dEds=tab.dEds_grid, meta=json.dumps(tab.meta))
        return tab

    @classmethod
    def from_config(cls, bands_cfg, cache_dir="results/cache"):
        return cls.load_or_compute(cache_dir, s_min=bands_cfg.s_min, s_max=bands_cfg.s_max,
                                   n_s=bands_cfg.n_s, n_bands=bands_cfg.n_bands,
                                   n_pw=bands_cfg.n_planewaves)

    # -- evaluation ----------------------------------------------------------------
    def _eval(self, name, s, n, deriv=0):
        s = np.asarray(s, float)
        n = np.asarray(n, np.int64)
        if np.any(s < self.s_grid[0] * (1 - 1e-12)) or np.any(s > self.s_grid[-1] * (1 + 1e-12)):
            raise ValueError(f"s outside table range [{self.s_grid[0]}, {self.s_grid[-1]}]: "
                             f"min {s.min():.3g}, max {s.max():.3g}")
        if np.any(n < 0) or np.any(n >= self.n_bands):
            raise ValueError("band index outside table")
        u = np.log(s)
        i = np.clip(((u - self.u_grid[0]) / self.du).astype(np.int64), 0, len(self.u_grid) - 2)
        t = u - self.u_grid[i]
        c = self._coef[name]
        c3, c2, c1, c0 = c[0, i, n], c[1, i, n], c[2, i, n], c[3, i, n]
        if deriv == 0:
            return ((c3 * t + c2) * t + c1) * t + c0
        # d/ds = (1/s) d/du
        return ((3 * c3 * t + 2 * c2) * t + c1) / s

    def energy(self, s, n):
        """Band-averaged energy E_n(s) (E_R, measured from the lattice maximum)."""
        return self._eval("E_mean", s, n)

    def dEds(self, s, n):
        """Hellmann-Feynman derivative dE_n/ds = -<cos^2>_n (dimensionless)."""
        return self._eval("dEds_grid", s, n)

    def edges(self, s, n):
        return self._eval("E_min", s, n), self._eval("E_max", s, n)

    def n_bound(self, s, tilt=0.0):
        """Number of bands whose upper edge lies below -tilt (tilt in E_R, >= 0)."""
        s = np.atleast_1d(np.asarray(s, float))
        nb = np.zeros(s.shape, np.int64)
        for n in range(self.n_bands):
            _, emax = self.edges(s, np.full(s.shape, n))
            nb += (emax < -tilt)
        return nb
