"""A laser beam = transverse mode (or mode superposition) + placement in the lab frame.

A ``Beam`` knows its power, the point where its waist sits (``origin``), its propagation
direction ``a`` (unit vector) and an orthonormal transverse frame (e1, e2) with
e2 = a x e1.  It converts lab positions r (shape (N, 3)) into beam coordinates
(x', y', s) and returns the complex field  E = sqrt(P) u  (|E|^2 = intensity) together
with its decomposition into an unwrapped reference phase and a residual envelope
(see ``modes.py``), and their gradients in the lab frame.

Lab frame: z vertical (up), z = 0 at the fiber facet, x-z the plane of the beam tilt.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .modes import Mode, build_mode


@dataclass
class BeamFieldData:
    """Field data at N points.  ``E = residual * exp(i phase_ref)`` (scaled by sqrt(P))."""
    residual: np.ndarray                 # complex (N,)
    phase_ref: np.ndarray                # real, unwrapped (N,)
    grad_residual: np.ndarray | None = None   # complex (N, 3), lab frame
    grad_phase_ref: np.ndarray | None = None  # real (N, 3), lab frame

    @property
    def intensity(self):
        return np.abs(self.residual) ** 2

    @property
    def field(self):
        return self.residual * np.exp(1j * self.phase_ref)

    @property
    def grad_intensity(self):
        """grad |E|^2 = 2 Re(conj(r) grad r)."""
        return 2.0 * np.real(np.conj(self.residual)[:, None] * self.grad_residual)


class Beam:
    def __init__(self, mode: Mode, power: float, direction, origin=(0.0, 0.0, 0.0),
                 transverse_hint=(1.0, 0.0, 0.0)):
        self.mode = mode
        self.power = float(power)
        a = np.asarray(direction, float)
        self.a = a / np.linalg.norm(a)
        self.origin = np.asarray(origin, float)
        h = np.asarray(transverse_hint, float)
        e1 = h - (h @ self.a) * self.a
        if np.linalg.norm(e1) < 1e-8:                 # hint parallel to the axis
            h = np.array([0.0, 1.0, 0.0])
            e1 = h - (h @ self.a) * self.a
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.a, self.e1)
        self.sqrtP = np.sqrt(self.power)

    # -- geometry ---------------------------------------------------------------
    @property
    def wavelength(self):
        return self.mode.wavelength

    @property
    def w0(self):
        return self.mode.w0

    @property
    def zR(self):
        return self.mode.zR

    def to_beam_frame(self, r):
        d = np.atleast_2d(r) - self.origin
        return d @ self.e1, d @ self.e2, d @ self.a

    def axis_point(self, s):
        """Lab position of the beam axis at propagation coordinate s."""
        return self.origin + np.multiply.outer(s, self.a)

    def radius_at(self, r):
        """Local 1/e^2 radius w(s) at lab points r."""
        _, _, s = self.to_beam_frame(r)
        return self.mode.w(s)

    # -- fields -----------------------------------------------------------------
    def field_data(self, r, grad: bool = False) -> BeamFieldData:
        xp, yp, s = self.to_beam_frame(r)
        res = self.sqrtP * self.mode.residual(xp, yp, s)
        ph = self.mode.ref_phase(xp, yp, s)
        if not grad:
            return BeamFieldData(res, ph)
        rg = self.mode.residual_grad(xp, yp, s)
        pg = self.mode.ref_phase_grad(xp, yp, s)
        basis = (self.e1, self.e2, self.a)
        g_res = self.sqrtP * sum(np.multiply.outer(np.broadcast_to(c, xp.shape), b)
                                 for c, b in zip(rg, basis))
        g_ph = sum(np.multiply.outer(np.broadcast_to(c, xp.shape), b) for c, b in zip(pg, basis))
        return BeamFieldData(res, ph, g_res, g_ph)

    def intensity(self, r):
        return self.field_data(r).intensity

    def field(self, r):
        return self.field_data(r).field

    def __repr__(self):
        return (f"Beam({self.mode.describe()}, P={self.power} W, w0={self.w0*1e6:.1f} um, "
                f"dir={np.round(self.a, 6)}, origin={self.origin})")


def make_beam_pair(optics_cfg):
    """Build (beam_up, beam_down) from an ``OpticsConfig``.

    beam_up   : exits the fiber, propagates along +z, waist at z = waist_z on the z axis.
    beam_down : propagates along -(sin theta, 0, cos theta); its axis passes through the
                point (0, 0, waist_z), so the two axes cross at the facet.
    """
    lam = optics_cfg.wavelength
    bu, bd = optics_cfg.beam_up, optics_cfg.beam_down
    th = optics_cfg.theta_rad
    up = Beam(build_mode(bu.modes, lam, bu.w0), bu.power, (0.0, 0.0, 1.0),
              origin=(0.0, 0.0, bu.waist_z))
    down = Beam(build_mode(bd.modes, lam, bd.w0), bd.power,
                (-np.sin(th), 0.0, -np.cos(th)), origin=(0.0, 0.0, bd.waist_z))
    return up, down
