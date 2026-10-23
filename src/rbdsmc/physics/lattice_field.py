"""Two-beam standing-wave field: light shifts, lattice depth, offset and site surfaces.

With parallel linear polarisations and U_i = c_U |E_i|^2 (c_U = alpha / (2 eps0 c)),

    V(r) = -U1 - U2 - 2 sqrt(U1 U2) cos(Phi) + m g z
         = V_off - U_lat cos^2(Phi/2) + m g z,
    U_lat = 4 sqrt(U1 U2),     V_off = -(sqrt(U1) - sqrt(U2))^2,
    Phi   = arg(E_down E_up^*) + phi_0    (unwrapped; increases with z).

On the common axis (theta = 0)  Phi = 2kz + k rho^2/R(z) - 2 zeta(z) + phi_0.
Lattice sites (intensity maxima, red detuning) are the surfaces Phi = 2 pi j; site j is
parametrised by its height z_j(x, y).  Everything here is independent of the transverse
mode content of the beams (see modes.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import G_ACC, M_RB, light_shift_coefficient
from .beams import Beam, make_beam_pair


@dataclass
class LatticeLocal:
    """Local lattice quantities at N points (SI units)."""
    U1: np.ndarray            # light shift magnitude of the upward beam (J, >= 0)
    U2: np.ndarray            # light shift magnitude of the downward beam (J, >= 0)
    Phi: np.ndarray           # unwrapped relative phase (rad)
    gU1: np.ndarray | None = None   # (N, 3)
    gU2: np.ndarray | None = None
    gPhi: np.ndarray | None = None

    @property
    def Ulat(self):
        return 4.0 * np.sqrt(self.U1 * self.U2)

    @property
    def Voff(self):
        return -(np.sqrt(self.U1) - np.sqrt(self.U2)) ** 2

    @property
    def Uavg(self):
        """Lattice-averaged optical potential  V_off - U_lat/2 = -(U1 + U2)."""
        return -(self.U1 + self.U2)

    @property
    def gUlat(self):
        s1, s2 = np.sqrt(self.U1)[:, None], np.sqrt(self.U2)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = 2.0 * (s2 / s1 * self.gU1 + s1 / s2 * self.gU2)
        return np.nan_to_num(g)

    @property
    def gVoff(self):
        s1, s2 = np.sqrt(self.U1)[:, None], np.sqrt(self.U2)[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            g = -(s1 - s2) * (self.gU1 / s1 - self.gU2 / s2)
        return np.nan_to_num(g)


class TwoBeamLattice:
    def __init__(self, beam_up: Beam, beam_down: Beam, alpha_au: float, gravity: bool = True,
                 phase_offset: float = 0.0, mass: float = M_RB):
        self.up, self.down = beam_up, beam_down
        self.cU = light_shift_coefficient(alpha_au)
        self.gravity = gravity
        self.phi0 = float(phase_offset)
        self.mass = mass
        self.mg = mass * G_ACC if gravity else 0.0
        self.wavelength = beam_up.wavelength
        self.k = 2.0 * np.pi / self.wavelength
        self.d = self.wavelength / 2.0

    @classmethod
    def from_config(cls, cfg):
        up, down = make_beam_pair(cfg.optics)
        return cls(up, down, cfg.optics.polarizability_au, cfg.physics.gravity,
                   cfg.optics.phase_offset_rad)

    # ------------------------------------------------------------------ local fields
    def local(self, r, grad: bool = False) -> LatticeLocal:
        r = np.atleast_2d(np.asarray(r, float))
        f1 = self.up.field_data(r, grad)
        f2 = self.down.field_data(r, grad)
        U1 = self.cU * f1.intensity
        U2 = self.cU * f2.intensity
        Phi = (f2.phase_ref - f1.phase_ref
               + np.angle(f2.residual * np.conj(f1.residual)) + self.phi0)
        if not grad:
            return LatticeLocal(U1, U2, Phi)
        gU1 = self.cU * f1.grad_intensity
        gU2 = self.cU * f2.grad_intensity
        gPhi = (f2.grad_phase_ref - f1.grad_phase_ref
                + _grad_arg(f2.residual, f2.grad_residual)
                - _grad_arg(f1.residual, f1.grad_residual))
        return LatticeLocal(U1, U2, Phi, gU1, gU2, gPhi)

    # ------------------------------------------------------------------ potentials
    def gravity_energy(self, r):
        return self.mg * np.atleast_2d(r)[:, 2]

    def potential(self, r):
        """Full potential V(r) including the lattice modulation and gravity (J)."""
        L = self.local(r)
        return -L.U1 - L.U2 - 2.0 * np.sqrt(L.U1 * L.U2) * np.cos(L.Phi) + self.gravity_energy(r)

    def potential_direct(self, r):
        """Same as ``potential`` but from |E1 + E2|^2 directly (for testing)."""
        E = self.up.field(r) + self.down.field(r) * np.exp(1j * self.phi0)
        return -self.cU * np.abs(E) ** 2 + self.gravity_energy(r)

    def potential_avg(self, r):
        """Lattice-averaged potential -(U1 + U2) + m g z (free atoms)."""
        L = self.local(r)
        return L.Uavg + self.gravity_energy(r)

    def force_avg(self, r):
        L = self.local(r, grad=True)
        F = L.gU1 + L.gU2
        F[:, 2] -= self.mg
        return F

    # ------------------------------------------------------------------ sites
    def site_index(self, r):
        """Index j of the nearest lattice site (Phi = 2 pi j)."""
        return np.rint(self.local(r).Phi / (2.0 * np.pi)).astype(np.int64)

    def site_surface(self, x, y, j, z_guess=None, tol=1e-10, max_iter=30):
        """Height z_j(x, y) of site surface j by Newton iteration on Phi = 2 pi j.

        Returns (z, LatticeLocal at the surface, with gradients)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.broadcast_to(np.asarray(y, float), x.shape).astype(float)
        j = np.broadcast_to(np.asarray(j), x.shape)
        if z_guess is None:
            z = j * self.d
        else:
            z = np.broadcast_to(np.asarray(z_guess, float), x.shape).astype(float).copy()
        target = 2.0 * np.pi * j
        for _ in range(max_iter):
            L = self.local(np.column_stack([x, y, z]), grad=True)
            dz = (L.Phi - target) / L.gPhi[:, 2]
            z = z - dz
            if np.max(np.abs(dz)) * self.k < tol:
                break
        else:
            raise RuntimeError("site_surface: Newton iteration did not converge")
        L = self.local(np.column_stack([x, y, z]), grad=True)
        return z, L

    def site_index_near(self, z):
        """Index of the on-axis site nearest to height z."""
        return self.site_index(np.array([[0.0, 0.0, z]]))[0] if np.ndim(z) == 0 else \
            self.site_index(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))


def _grad_arg(res, gres):
    """grad arg(res) = Im(conj(res) grad res) / |res|^2 (zero where res = 0)."""
    a2 = np.abs(res) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.imag(np.conj(res)[:, None] * gres) / a2[:, None]
    return np.nan_to_num(g)
