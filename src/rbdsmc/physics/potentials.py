"""Semiclassical trap: band-resolved radial potentials for bound atoms, lattice-averaged
potential for free atoms, release criterion, energy bookkeeping and the fiber test.

Bound atom (site j, band n, transverse position (x, y) on the site surface z_j(x, y)):

    V_n(x, y; j) = V_off + E_R E_n(s) + m g z_j(x, y),      s = U_lat / E_R,

evaluated on the surface.  The radial force is the gradient along the surface,

    dV/dx = (dV/dx)_3D + (dV/dz)_3D dz_j/dx,   dz_j/dx = -dPhi/dx / dPhi/dz,
    grad_3D V = grad V_off + (dE_n/ds) grad U_lat + m g z_hat.

(The kinetic energy of the slow vertical motion along the curved surface, a relative
correction ~ (slope)^2 < 1e-4, is neglected.)

Band n is bound if its upper edge lies below the downhill barrier:  E_n^max(s) < -tilt,
tilt = m g d / (2 E_R).  When an atom's band stops being bound it is converted into a free
particle with  1/2 m v_z^2 = E_R E_n + U_lat/2  (energy conserving, random sign).

Free atom:  V_free(r) = -(U1 + U2) + m g z  (lattice average).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import KB, M_RB, recoil_energy
from .bands import BandTable
from .lattice_field import TwoBeamLattice


@dataclass
class BoundEval:
    """Evaluation of bound-atom quantities at N points."""
    V: np.ndarray          # V_n (J)
    F: np.ndarray | None   # radial force (N, 2) (J/m)
    z: np.ndarray          # site-surface height z_j(x, y)
    s: np.ndarray          # U_lat / E_R (clipped to the table range)
    Ulat: np.ndarray
    E_band: np.ndarray     # E_R * E_n(s)   (J, measured from the lattice top)
    bound: np.ndarray      # bool: band still bound at this point


class SemiclassicalTrap:
    def __init__(self, cfg, lattice: TwoBeamLattice | None = None,
                 bands: BandTable | None = None, cache_dir="results/cache"):
        self.cfg = cfg
        self.lat = lattice or TwoBeamLattice.from_config(cfg)
        self.tab = bands or BandTable.from_config(cfg.bands, cache_dir=cache_dir)
        self.m = M_RB
        self.ER = recoil_energy(cfg.optics.wavelength)
        self.d = self.lat.d
        self.mg = self.lat.mg
        self.tilt = 0.5 * self.mg * self.d / self.ER          # E_R units
        self.s_lo, self.s_hi = self.tab.s_grid[0], self.tab.s_grid[-1]
        self.theta = cfg.optics.theta_rad
        self.z_c = cfg.cloud.z_center
        self.z_half = cfg.region.z_half
        self.rho_factor = cfg.region.rho_factor
        self.w0 = self.lat.up.w0
        self.a_core = cfg.fiber.core_radius(self.w0)
        # averaged in-fiber potential depth on the axis at the facet (U1 + U2)
        Lf = self.lat.local(np.array([[0.0, 0.0, 0.0]]))
        self.U_fiber = float(Lf.U1[0] + Lf.U2[0])
        # reference energy: classical minimum of V at the central site
        self.j_c = int(self.lat.site_index(np.array([[self.x_center(self.z_c), 0.0, self.z_c]]))[0])
        zc, Lc = self.lat.site_surface(np.array([self.x_center(self.z_c)]), np.array([0.0]), self.j_c)
        self.E_ref = float(Lc.Voff[0] - Lc.Ulat[0] + self.mg * zc[0])
        self.z_site_c = float(zc[0])
        # per-site reference energies (classical minimum of each site) over the trap region
        margin = 0.1e-3
        j_lo = int(self.site_index_at(self.z_c - self.z_half - margin)[0])
        j_hi = int(self.site_index_at(self.z_c + self.z_half + margin)[0])
        self._jref = np.arange(j_lo, j_hi + 1)
        zj = self.site_z_axis(self._jref)
        zj, Lj = self.lat.site_surface(self.x_center(zj), np.zeros_like(zj), self._jref, z_guess=zj)
        self._Eref = Lj.Voff - Lj.Ulat + self.mg * zj

    # ------------------------------------------------------------------ geometry
    def x_center(self, z):
        """x of the lattice centre line (bisector of the two beam axes) at height z."""
        return np.asarray(z) * np.tan(self.theta) / 2.0

    def beam_radius(self, z):
        return self.lat.up.mode.w(np.asarray(z, float))

    def rho_region(self, z):
        """Transverse radius of the trap region at height z."""
        return self.rho_factor * self.beam_radius(z)

    def rho_from_center(self, x, y, z):
        return np.hypot(x - self.x_center(z), y)

    def site_z_axis(self, j):
        """Height of site j on the lattice centre line (Newton)."""
        j = np.atleast_1d(j)
        z0 = j * self.d
        z, _ = self.lat.site_surface(self.x_center(z0), np.zeros_like(z0, dtype=float), j, z_guess=z0)
        return z

    def site_index_at(self, z):
        z = np.atleast_1d(np.asarray(z, float))
        return self.lat.site_index(np.column_stack([self.x_center(z), np.zeros_like(z), z]))

    def site_reference_energy(self, j):
        """Classical minimum of V in site j (J): V_off - U_lat + m g z on the centre line.
        Energies 'in the trap' are measured from this per-site reference."""
        j = np.asarray(j, np.int64)
        k = j - self._jref[0]
        if np.any(k < 0) or np.any(k >= len(self._jref)):
            raise ValueError("site outside the tabulated trap region")
        return self._Eref[k]

    # ------------------------------------------------------------------ bands
    def _s(self, Ulat):
        return np.clip(Ulat / self.ER, self.s_lo, self.s_hi)

    def band_is_bound(self, s, n):
        _, emax = self.tab.edges(s, n)
        return (emax < -self.tilt) & (s > self.s_lo)

    def axial_energies(self, s, n):
        """(E_band, <T_z>, <V_lat>) in J for band n at depth s; E_band from the lattice top."""
        E = self.tab.energy(s, n)
        dE = self.tab.dEds(s, n)
        return self.ER * E, self.ER * (E - s * dE), self.ER * s * dE

    # ------------------------------------------------------------------ bound atoms
    def bound_eval(self, x, y, j, n, z_guess=None, force: bool = True) -> BoundEval:
        z, L = self.lat.site_surface(x, y, j, z_guess=z_guess)
        s = self._s(L.Ulat)
        n = np.asarray(n, np.int64)
        Eb = self.ER * self.tab.energy(s, n)
        V = L.Voff + Eb + self.mg * z
        bound = self.band_is_bound(s, n)
        F = None
        if force:
            dEds = self.tab.dEds(s, n)
            g3 = L.gVoff + dEds[:, None] * L.gUlat
            g3[:, 2] += self.mg
            dzdx = -L.gPhi[:, 0] / L.gPhi[:, 2]
            dzdy = -L.gPhi[:, 1] / L.gPhi[:, 2]
            F = -np.column_stack([g3[:, 0] + g3[:, 2] * dzdx, g3[:, 1] + g3[:, 2] * dzdy])
        return BoundEval(V, F, z, s, L.Ulat, Eb, bound)

    def bound_energy(self, x, y, vx, vy, j, n, z_guess=None):
        e = self.bound_eval(x, y, j, n, z_guess, force=False)
        return 0.5 * self.m * (vx * vx + vy * vy) + e.V

    def release_vz2(self, E_band, Ulat):
        """v_z^2 of an atom converted from a band (energy E_band from the lattice top) to a
        free particle in the lattice-averaged potential."""
        return np.maximum(2.0 * (E_band + 0.5 * Ulat) / self.m, 0.0)

    # ------------------------------------------------------------------ free atoms
    def free_potential(self, r):
        return self.lat.potential_avg(r)

    def free_force(self, r):
        return self.lat.force_avg(r)

    def free_energy(self, r, v):
        return 0.5 * self.m * np.einsum("ij,ij->i", v, v) + self.free_potential(r)

    # ------------------------------------------------------------------ fiber
    def fiber_radially_bound(self, rho, vperp2):
        """Transverse turning point inside the core of the (lattice-averaged) guided
        potential -U_f exp(-2 rho^2 / w0^2), U_f = U1 + U2 at the facet."""
        Ef = 0.5 * self.m * vperp2 - self.U_fiber * np.exp(-2.0 * rho ** 2 / self.w0 ** 2)
        E_wall = -self.U_fiber * np.exp(-2.0 * self.a_core ** 2 / self.w0 ** 2)
        return (rho < self.a_core) & (Ef < E_wall), Ef

    # ------------------------------------------------------------------ summary
    def describe(self) -> dict:
        from ..constants import UK
        L = self.lat.local(np.array([[self.x_center(self.z_c), 0.0, self.z_site_c]]))
        return {
            "E_R_nK": self.ER / KB * 1e9, "tilt_ER": self.tilt,
            "Ulat_center_uK": float(L.Ulat[0] / UK), "site_center": self.j_c,
            "E_ref_center_uK": self.E_ref / UK, "U_fiber_uK": self.U_fiber / UK,
            "a_core_um": self.a_core * 1e6,
        }
