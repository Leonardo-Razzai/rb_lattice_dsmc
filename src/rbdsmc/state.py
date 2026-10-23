"""Particle state: fixed-size structure-of-arrays.  Atoms are never removed; their status
changes, so every atom keeps its full history (loss time, channel, fiber-entry data).

Status codes
------------
BOUND        in a lattice site and band (semiclassical: j, n, x, y, vx, vy)
FREE         axially unbound, inside the trap region (classical 3D: x, y, z, vx, vy, vz)
IN_TRANSIT   fell below the trap region, not yet at the facet (lost from the trap)
LOST_RADIAL  left the trap region transversely
LOST_FACET   reached z = 0 outside the fiber core
IN_FIBER     reached z = 0 inside the core (see ``fiber_bound``)
LOST_OTHER   one-/three-body losses (optional)
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import IntEnum

import numpy as np


class Status(IntEnum):
    BOUND = 0
    FREE = 1
    IN_TRANSIT = 2
    LOST_RADIAL = 3
    LOST_FACET = 4
    IN_FIBER = 5
    LOST_OTHER = 6


IN_REGION = (Status.BOUND, Status.FREE)
PROPAGATED = (Status.BOUND, Status.FREE, Status.IN_TRANSIT)


@dataclass
class ParticleState:
    status: np.ndarray        # int8
    j: np.ndarray             # int64 site index
    n: np.ndarray             # int16 band index (meaningful when BOUND)
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray             # site-surface height (BOUND) or position (FREE, IN_TRANSIT)
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray            # 0 for BOUND
    ax: np.ndarray            # cached accelerations
    ay: np.ndarray
    az: np.ndarray
    n_coll: np.ndarray        # int32 collision counter
    t_event: np.ndarray       # time of the last status change (s); nan if never
    E0: np.ndarray            # energy at t = 0 (J), for diagnostics
    released_band: np.ndarray # int16, band from which the atom was released (-1 if never)
    fiber_bound: np.ndarray   # bool, radially bound when entering the fiber
    entry_rho: np.ndarray     # rho at z = 0 (nan if not arrived)
    entry_vperp: np.ndarray
    entry_Eperp: np.ndarray

    @classmethod
    def empty(cls, N: int) -> "ParticleState":
        f = lambda dt=float, val=0.0: np.full(N, val, dtype=dt)     # noqa: E731
        return cls(status=f(np.int8, 0), j=f(np.int64, 0), n=f(np.int16, 0),
                   x=f(), y=f(), z=f(), vx=f(), vy=f(), vz=f(), ax=f(), ay=f(), az=f(),
                   n_coll=f(np.int32, 0), t_event=f(float, np.nan), E0=f(float, np.nan),
                   released_band=f(np.int16, -1), fiber_bound=f(bool, False),
                   entry_rho=f(float, np.nan), entry_vperp=f(float, np.nan),
                   entry_Eperp=f(float, np.nan))

    @property
    def N(self) -> int:
        return len(self.status)

    def mask(self, *statuses) -> np.ndarray:
        return np.isin(self.status, [int(s) for s in statuses])

    def counts(self) -> dict:
        c = np.bincount(self.status.astype(np.int64), minlength=len(Status))
        return {s.name: int(c[s]) for s in Status}

    def r(self, idx=None) -> np.ndarray:
        idx = slice(None) if idx is None else idx
        return np.column_stack([self.x[idx], self.y[idx], self.z[idx]])

    def v(self, idx=None) -> np.ndarray:
        idx = slice(None) if idx is None else idx
        return np.column_stack([self.vx[idx], self.vy[idx], self.vz[idx]])

    def copy(self) -> "ParticleState":
        return ParticleState(**{f.name: getattr(self, f.name).copy() for f in fields(self)})

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
