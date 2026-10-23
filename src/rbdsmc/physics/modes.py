"""Transverse optical modes in a beam's own frame (x', y' transverse, s along propagation).

Convention (Siegman): a mode of Gouy order N propagating along +s is

    u(x', y', s) = residual(x', y', s) * exp(i * phi_ref(x', y', s)),
    phi_ref      = -k s - k rho^2 / (2 R(s)) + zeta(s)            (TEM00 phase),

with zeta = arctan(s/z_R), 1/R = s/(s^2 + z_R^2), all valid for s < 0 (before the waist).
``phi_ref`` is the large, smooth, *unwrapped* phase shared by all modes of the same
waist and wavelength.  The ``residual`` is a slowly varying complex envelope carrying the
transverse amplitude pattern and the extra Gouy phase N*zeta; for TEM00 it is real and
positive.  Splitting the phase this way lets the lattice code count lattice sites with an
unwrapped relative phase while staying agnostic of the mode content.

Normalisation: integral |u|^2 dx' dy' = 1 at every s, so the intensity of a beam of power
P is I = P |u|^2.

Implemented modes: GaussianTEM00 (analytic gradients), LaguerreGauss(p, l),
HermiteGauss(m, n) (finite-difference gradients unless overridden), and ModeSum, a
normalised coherent superposition.  New modes (e.g. projections of hollow-core-fiber EH
modes) only need to implement ``residual`` (and optionally ``residual_grad``).
"""
from __future__ import annotations

from math import factorial

import numpy as np
from scipy.special import eval_genlaguerre, eval_hermite


class Mode:
    """Base class: geometry of a Gaussian-beam family with waist w0 at s = 0."""

    gouy_order = 0     # N; ModeSum overrides with None (mixed orders)

    def __init__(self, wavelength: float, w0: float):
        self.wavelength = float(wavelength)
        self.w0 = float(w0)
        self.k = 2.0 * np.pi / self.wavelength
        self.zR = np.pi * self.w0 ** 2 / self.wavelength

    # -- Gaussian-beam geometry ---------------------------------------------
    def w(self, s):
        return self.w0 * np.sqrt(1.0 + (s / self.zR) ** 2)

    def inv_R(self, s):
        """1/R(s) = s / (s^2 + z_R^2); finite at the waist."""
        return s / (s * s + self.zR ** 2)

    def gouy(self, s):
        return np.arctan(s / self.zR)

    def ref_phase(self, xp, yp, s):
        rho2 = xp * xp + yp * yp
        return -self.k * s - 0.5 * self.k * rho2 * self.inv_R(s) + self.gouy(s)

    def ref_phase_grad(self, xp, yp, s):
        rho2 = xp * xp + yp * yp
        d = s * s + self.zR ** 2
        iR = s / d
        diR_ds = (self.zR ** 2 - s * s) / d ** 2
        gx = -self.k * xp * iR
        gy = -self.k * yp * iR
        gs = -self.k - 0.5 * self.k * rho2 * diR_ds + self.zR / d
        return gx, gy, gs

    # -- mode-specific part --------------------------------------------------
    def residual(self, xp, yp, s):
        raise NotImplementedError

    def residual_grad(self, xp, yp, s):
        """Gradient (d/dx', d/dy', d/ds) of the residual; central finite differences by
        default (step 1e-4 of the local beam radius, 1e-4 z_R along s)."""
        hx = 1e-4 * self.w(s)
        hs = 1e-4 * self.zR
        gx = (self.residual(xp + hx, yp, s) - self.residual(xp - hx, yp, s)) / (2 * hx)
        gy = (self.residual(xp, yp + hx, s) - self.residual(xp, yp - hx, s)) / (2 * hx)
        gs = (self.residual(xp, yp, s + hs) - self.residual(xp, yp, s - hs)) / (2 * hs)
        return gx, gy, gs

    # -- full field ------------------------------------------------------------
    def field(self, xp, yp, s):
        return self.residual(xp, yp, s) * np.exp(1j * self.ref_phase(xp, yp, s))

    def field_grad(self, xp, yp, s):
        r = self.residual(xp, yp, s)
        rg = self.residual_grad(xp, yp, s)
        pg = self.ref_phase_grad(xp, yp, s)
        e = np.exp(1j * self.ref_phase(xp, yp, s))
        return tuple((rgi + 1j * r * pgi) * e for rgi, pgi in zip(rg, pg))

    def describe(self) -> str:
        return type(self).__name__


class GaussianTEM00(Mode):
    gouy_order = 0

    def residual(self, xp, yp, s):
        w = self.w(s)
        rho2 = xp * xp + yp * yp
        return (np.sqrt(2.0 / np.pi) / w * np.exp(-rho2 / (w * w))).astype(complex)

    def residual_grad(self, xp, yp, s):
        w = self.w(s)
        r = self.residual(xp, yp, s)
        rho2 = xp * xp + yp * yp
        d = s * s + self.zR ** 2
        gx = -2.0 * xp / w ** 2 * r
        gy = -2.0 * yp / w ** 2 * r
        gs = (-s / d + 2.0 * rho2 * s / (w ** 2 * d)) * r
        return gx, gy, gs


class LaguerreGauss(Mode):
    """LG_{p,l}: N = 2p + |l|."""

    def __init__(self, wavelength, w0, p: int = 0, l: int = 0):
        super().__init__(wavelength, w0)
        self.p, self.l = int(p), int(l)
        self.gouy_order = 2 * self.p + abs(self.l)
        al = abs(self.l)
        self.norm = np.sqrt(2.0 * factorial(self.p) / (np.pi * factorial(self.p + al)))

    def residual(self, xp, yp, s):
        w = self.w(s)
        rho2 = xp * xp + yp * yp
        x = 2.0 * rho2 / (w * w)
        al = abs(self.l)
        radial = self.norm / w * x ** (al / 2.0) * eval_genlaguerre(self.p, al, x) * np.exp(-x / 2)
        phase = self.l * np.arctan2(yp, xp) + self.gouy_order * self.gouy(s)
        return radial * np.exp(1j * phase)

    def describe(self):
        return f"LG(p={self.p},l={self.l})"


class HermiteGauss(Mode):
    """HG_{m,n}: N = m + n."""

    def __init__(self, wavelength, w0, m: int = 0, n: int = 0):
        super().__init__(wavelength, w0)
        self.m, self.n = int(m), int(n)
        self.gouy_order = self.m + self.n
        self.norm = np.sqrt(2.0 / np.pi) / np.sqrt(2.0 ** (self.m + self.n)
                                                  * factorial(self.m) * factorial(self.n))

    def residual(self, xp, yp, s):
        w = self.w(s)
        rho2 = xp * xp + yp * yp
        a = np.sqrt(2.0) / w
        amp = (self.norm / w * eval_hermite(self.m, a * xp) * eval_hermite(self.n, a * yp)
               * np.exp(-rho2 / (w * w)))
        return amp * np.exp(1j * self.gouy_order * self.gouy(s))

    def describe(self):
        return f"HG(m={self.m},n={self.n})"


class ModeSum(Mode):
    """Coherent superposition sum_i c_i u_i of modes sharing wavelength and waist.

    Coefficients are normalised so that sum |c_i|^2 = 1 (exact unit power when the modes
    are orthonormal, which holds within the LG or within the HG family)."""

    gouy_order = None

    def __init__(self, modes, coeffs):
        m0 = modes[0]
        super().__init__(m0.wavelength, m0.w0)
        for m in modes:
            if abs(m.w0 - self.w0) > 1e-12 * self.w0 or abs(m.wavelength - self.wavelength) > 1e-15:
                raise ValueError("all modes in a ModeSum must share wavelength and waist")
        c = np.asarray(coeffs, dtype=complex)
        self.coeffs = c / np.sqrt(np.sum(np.abs(c) ** 2))
        self.modes = list(modes)
        orders = {m.gouy_order for m in modes}
        if len(orders) == 1:
            self.gouy_order = orders.pop()

    def residual(self, xp, yp, s):
        return sum(c * m.residual(xp, yp, s) for c, m in zip(self.coeffs, self.modes))

    def residual_grad(self, xp, yp, s):
        gx = gy = gs = 0.0
        for c, m in zip(self.coeffs, self.modes):
            a, b, d = m.residual_grad(xp, yp, s)
            gx, gy, gs = gx + c * a, gy + c * b, gs + c * d
        return gx, gy, gs

    def describe(self):
        return " + ".join(f"({c:.3g}){m.describe()}" for c, m in zip(self.coeffs, self.modes))


def build_mode(specs, wavelength: float, w0: float) -> Mode:
    """Build a mode from a config list such as
    [{"type": "TEM00"}] or [{"type": "LG", "p": 0, "l": 0, "coeff": [0.99, 0]},
                            {"type": "LG", "p": 1, "l": 0, "coeff": [0.1, 0]}]."""
    specs = list(specs)
    built, coeffs = [], []
    for sp in specs:
        t = sp.get("type", "TEM00").upper()
        if t == "TEM00":
            m = GaussianTEM00(wavelength, w0)
        elif t == "LG":
            m = LaguerreGauss(wavelength, w0, sp.get("p", 0), sp.get("l", 0))
        elif t == "HG":
            m = HermiteGauss(wavelength, w0, sp.get("m", 0), sp.get("n", 0))
        else:
            raise ValueError(f"unknown mode type {t!r}")
        c = sp.get("coeff", [1.0, 0.0])
        built.append(m)
        coeffs.append(complex(c[0], c[1]) if isinstance(c, (list, tuple)) else complex(c))
    if len(built) == 1 and abs(coeffs[0] - 1.0) < 1e-15:
        return built[0]
    return ModeSum(built, coeffs)
