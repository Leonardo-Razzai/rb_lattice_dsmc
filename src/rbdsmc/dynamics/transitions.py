"""Status changes: bound -> free release, trap-region exits, arrival at the fiber facet."""
from __future__ import annotations

import numpy as np

from ..state import ParticleState, Status


def release_to_free(trap, st: ParticleState, idx, E_band, Ulat, z, t, rng):
    """Convert BOUND atoms ``idx`` to FREE, conserving energy: 1/2 m vz^2 = E_band + U_lat/2."""
    if len(idx) == 0:
        return
    vz = np.sqrt(trap.release_vz2(E_band, Ulat))
    sign = np.where(rng.random(len(idx)) < 0.5, -1.0, 1.0)
    st.z[idx] = z
    st.vz[idx] = sign * vz
    st.released_band[idx] = st.n[idx]
    st.status[idx] = Status.FREE
    st.t_event[idx] = t
    F = trap.free_force(st.r(idx))
    st.ax[idx], st.ay[idx], st.az[idx] = (F / trap.m).T


def classify_free_exits(trap, st: ParticleState, idx, t):
    """FREE atoms below the trap region -> IN_TRANSIT; outside it transversely -> LOST_RADIAL."""
    if len(idx) == 0:
        return 0, 0
    z = st.z[idx]
    below = z < trap.z_c - trap.z_half
    rho = trap.rho_from_center(st.x[idx], st.y[idx], z)
    out = (~below) & (rho > trap.rho_region(z))
    st.status[idx[below]] = Status.IN_TRANSIT
    st.status[idx[out]] = Status.LOST_RADIAL
    st.t_event[idx[below | out]] = t
    return int(below.sum()), int(out.sum())


def classify_bound_exits(trap, st: ParticleState, idx, t):
    """BOUND atoms beyond the trap-region radius -> LOST_RADIAL (rare: release comes first)."""
    if len(idx) == 0:
        return 0
    rho = trap.rho_from_center(st.x[idx], st.y[idx], st.z[idx])
    out = rho > trap.rho_region(st.z[idx])
    st.status[idx[out]] = Status.LOST_RADIAL
    st.t_event[idx[out]] = t
    return int(out.sum())


def arrive_at_facet(trap, st: ParticleState, idx, r_old, v_old, r_new, v_new, t_old, dt):
    """Atoms that crossed z = 0 during a sub-step: interpolate to z = 0 and classify."""
    if len(idx) == 0:
        return 0, 0
    f = r_old[:, 2] / (r_old[:, 2] - r_new[:, 2])          # fraction of the sub-step
    f = np.clip(f, 0.0, 1.0)[:, None]
    r0 = r_old + f * (r_new - r_old)
    v0 = v_old + f * (v_new - v_old)
    rho = np.hypot(r0[:, 0], r0[:, 1])                       # fiber axis = z axis
    vperp2 = v0[:, 0] ** 2 + v0[:, 1] ** 2
    bound, Eperp = trap.fiber_radially_bound(rho, vperp2)
    in_core = rho < trap.a_core
    st.x[idx], st.y[idx], st.z[idx] = r0.T
    st.vx[idx], st.vy[idx], st.vz[idx] = v0.T
    st.status[idx] = np.where(in_core, Status.IN_FIBER, Status.LOST_FACET)
    st.fiber_bound[idx] = bound
    st.entry_rho[idx] = rho
    st.entry_vperp[idx] = np.sqrt(vperp2)
    st.entry_Eperp[idx] = Eperp
    st.t_event[idx] = t_old + f[:, 0] * dt
    return int(in_core.sum()), int((~in_core).sum())
