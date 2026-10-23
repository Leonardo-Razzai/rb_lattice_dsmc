"""Time stepping without collisions.

* BOUND atoms: 2D velocity Verlet in V_n(x, y; j) with step dt.  After the step, atoms whose
  band is no longer bound are released (energy-conserving) and become FREE.
* FREE and IN_TRANSIT atoms: 3D velocity Verlet in the lattice-averaged potential with
  ``substeps`` sub-steps per dt (radial frequencies reach ~2 kHz near the facet).

The step is symplectic for each atom between status changes, so energies are conserved to
O(dt^2) without secular drift.
"""
from __future__ import annotations

import numpy as np

from ..state import ParticleState, Status
from . import transitions as tr


def verlet(x, v, a, accel, dt, nsteps=1):
    """Generic velocity Verlet (used in tests): returns x, v, a after nsteps."""
    for _ in range(nsteps):
        v = v + 0.5 * dt * a
        x = x + dt * v
        a = accel(x)
        v = v + 0.5 * dt * a
    return x, v, a


class Propagator:
    def __init__(self, trap, dt: float, substeps: int, rng):
        self.trap = trap
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.rng = rng
        self.m = trap.m

    # ------------------------------------------------------------------ setup
    def init_accelerations(self, st: ParticleState):
        b = np.nonzero(st.status == Status.BOUND)[0]
        if len(b):
            e = self.trap.bound_eval(st.x[b], st.y[b], st.j[b], st.n[b], z_guess=st.z[b])
            st.z[b] = e.z
            st.ax[b], st.ay[b] = (e.F / self.m).T
            st.az[b] = 0.0
            # atoms sampled in a band that is not bound here are released immediately
            rel = b[~e.bound]
            if len(rel):
                tr.release_to_free(self.trap, st, rel, e.E_band[~e.bound], e.Ulat[~e.bound],
                                   e.z[~e.bound], 0.0, self.rng)
        f = np.nonzero(np.isin(st.status, [Status.FREE, Status.IN_TRANSIT]))[0]
        if len(f):
            F = self.trap.free_force(st.r(f))
            st.ax[f], st.ay[f], st.az[f] = (F / self.m).T

    # ------------------------------------------------------------------ one step
    def step(self, st: ParticleState, t: float) -> dict:
        """Advance from t to t + dt.  Returns counts of the status changes."""
        ev = {"released": 0, "lost_radial": 0, "to_transit": 0, "in_fiber": 0, "facet": 0}
        self._step_bound(st, t, ev)
        self._step_free(st, t, ev)
        return ev

    def _step_bound(self, st, t, ev):
        dt = self.dt
        b = np.nonzero(st.status == Status.BOUND)[0]
        if len(b) == 0:
            return
        st.vx[b] += 0.5 * dt * st.ax[b]
        st.vy[b] += 0.5 * dt * st.ay[b]
        st.x[b] += dt * st.vx[b]
        st.y[b] += dt * st.vy[b]
        e = self.trap.bound_eval(st.x[b], st.y[b], st.j[b], st.n[b], z_guess=st.z[b])
        st.z[b] = e.z
        st.ax[b], st.ay[b] = (e.F / self.m).T
        st.vx[b] += 0.5 * dt * st.ax[b]
        st.vy[b] += 0.5 * dt * st.ay[b]
        rel = ~e.bound
        if rel.any():
            tr.release_to_free(self.trap, st, b[rel], e.E_band[rel], e.Ulat[rel], e.z[rel],
                               t + dt, self.rng)
            ev["released"] += int(rel.sum())
        still = b[~rel]
        ev["lost_radial"] += tr.classify_bound_exits(self.trap, st, still, t + dt)

    def _step_free(self, st, t, ev):
        idx = np.nonzero(np.isin(st.status, [Status.FREE, Status.IN_TRANSIT]))[0]
        if len(idx) == 0:
            return
        h = self.dt / self.substeps
        r = st.r(idx)
        v = st.v(idx)
        a = np.column_stack([st.ax[idx], st.ay[idx], st.az[idx]])
        active = np.ones(len(idx), bool)
        for k in range(self.substeps):
            ai = np.nonzero(active)[0]
            if len(ai) == 0:
                break
            r_old, v_old = r[ai].copy(), v[ai].copy()
            v[ai] += 0.5 * h * a[ai]
            r[ai] += h * v[ai]
            a[ai] = self.trap.free_force(r[ai]) / self.m
            v[ai] += 0.5 * h * a[ai]
            hit = r[ai, 2] <= 0.0
            if hit.any():
                hi = ai[hit]
                n_in, n_out = tr.arrive_at_facet(self.trap, st, idx[hi], r_old[hit], v_old[hit],
                                                 r[hi], v[hi], t + k * h, h)
                ev["in_fiber"] += n_in
                ev["facet"] += n_out
                active[hi] = False
        keep = idx[active]
        st.x[keep], st.y[keep], st.z[keep] = r[active].T
        st.vx[keep], st.vy[keep], st.vz[keep] = v[active].T
        st.ax[keep], st.ay[keep], st.az[keep] = a[active].T
        free = keep[st.status[keep] == Status.FREE]
        n_tr, n_out = tr.classify_free_exits(self.trap, st, free, t + self.dt)
        ev["to_transit"] += n_tr
        ev["lost_radial"] += n_out
