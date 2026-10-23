"""Observables recorded during a run (definitions in docs/theory, section "Observables").

Atom sets (all fractions are relative to the initial number of test particles N):
  remaining  R : status BOUND or FREE (inside the trap region)
  bound      B : BOUND and E < E_esc(j, n)  (cannot leave without collisions)
  unbound      : R minus B
Energies are per atom, measured from the classical minimum of the atom's own site
(``trap.site_reference_energy(j)``; for free atoms the site they were released from), and
stored in microkelvin (E / k_B).

Bound atom:  E_kin = 1/2 m v_perp^2 + <T_z>_n,   E_tot = 1/2 m v_perp^2 + V_n(x, y; j)
Free atom:   E_kin = 1/2 m v^2,                  E_tot = E_kin + V_free(r)
Virial (set B):  < 2 K_perp - (r_perp - r_0) . grad V_n >, r_0 = lattice centre line;
                 the axial term vanishes identically for band eigenstates.
                 ratio = <2 K_perp> / <(r_perp - r_0) . grad V_n>  (1 in a stationary state).
T_perp : m <|v_perp - <v_perp>|^2> / (2 k_B) over B (and over R).
T_par  : temperature of a thermal band population (bound bands at the central depth)
         with the same mean band-excitation energy as set B.
T_kin  : 2 <E_kin> / (3 k_B) over B (includes the axial zero-point kinetic energy).
"""
from __future__ import annotations

import numpy as np

from ..constants import KB, UK
from ..state import Status


def thermal_band_temperature(eps, target):
    """T such that the Boltzmann mean of excitation energies ``eps`` (J) equals ``target``."""
    eps = np.asarray(eps, float)
    if target <= 0 or len(eps) < 2:
        return 0.0 if target <= 0 else np.nan
    if target >= eps.mean():
        return np.nan                                  # hotter than infinite T on a finite ladder

    def mean_at(T):
        w = np.exp(-(eps - eps.min()) / (KB * T))
        return np.sum(eps * w) / np.sum(w)

    lo, hi = np.log(1e-10), np.log(1e-2)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mean_at(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


class ObservableRecorder:
    def __init__(self, trap, escape, N0_real, kT0, n_band_max=20):
        self.trap = trap
        self.escape = escape
        self.N0_real = float(N0_real)
        self.kT0 = kT0
        self.n_band_max = n_band_max
        # reference band ladder at the central depth for T_par
        s_c = float(trap._s(np.array([trap.lat.local(np.array(
            [[trap.x_center(trap.z_c), 0.0, trap.z_site_c]])).Ulat[0]]))[0])
        self.s_c = s_c
        self.Ulat_c = s_c * trap.ER
        nb = int(trap.tab.n_bound(np.array([s_c]), trap.tilt)[0])
        n = np.arange(nb)
        E = trap.tab.energy(np.full(nb, s_c), n) * trap.ER
        self.eps_ref = E - E[0]

    def record(self, t, st, E0, n_coll_interval=0, interval=None):
        T = self.trap
        m = T.m
        N = st.N
        c = st.counts()
        row = {"t_ms": t * 1e3}
        # ------------------------------------------------------------ fractions
        b = np.nonzero(st.status == Status.BOUND)[0]
        f = np.nonzero(st.status == Status.FREE)[0]
        row["N_region"] = (len(b) + len(f)) / N * self.N0_real
        row["f_remaining"] = (len(b) + len(f)) / N
        row["f_band"] = len(b) / N                        # in a bound band (any energy)
        row["f_free_in_region"] = len(f) / N
        for key, s in [("f_lost_radial", "LOST_RADIAL"), ("f_in_transit", "IN_TRANSIT"),
                       ("f_facet", "LOST_FACET"), ("f_fiber", "IN_FIBER"),
                       ("f_lost_other", "LOST_OTHER")]:
            row[key] = c[s] / N
        row["f_fiber_bound"] = np.count_nonzero(st.fiber_bound) / N

        # ------------------------------------------------------------ energies
        Etot = np.empty(0)
        Ekin = np.empty(0)
        eb = np.zeros(len(b), bool)
        s_b = np.empty(0)
        if len(b):
            e = T.bound_eval(st.x[b], st.y[b], st.j[b], st.n[b], z_guess=st.z[b], force=False)
            _, Tz, _ = T.axial_energies(e.s, st.n[b])
            Kp = 0.5 * m * (st.vx[b] ** 2 + st.vy[b] ** 2)
            Eb_abs = Kp + e.V
            Eb_tot = Eb_abs - T.site_reference_energy(st.j[b])
            Eb_kin = Kp + Tz
            eb = Eb_abs < self.escape(st.j[b], st.n[b])
            s_b = e.s
            Etot, Ekin = Eb_tot, Eb_kin
        if len(f):
            v = st.v(f)
            Kf = 0.5 * m * np.einsum("ij,ij->i", v, v)
            Ef_abs = Kf + T.free_potential(st.r(f))
            Ef_tot = Ef_abs - T.site_reference_energy(st.j[f])
            Etot = np.concatenate([Etot, Ef_tot])
            Ekin = np.concatenate([Ekin, Kf])
        B = b[eb]
        row["f_bound"] = len(B) / N
        row["f_unbound"] = row["f_remaining"] - row["f_bound"]

        def stats(prefix, Et, Ek):
            if len(Et) == 0:
                for k in ("E_tot", "E_kin", "E_pot"):
                    row[f"{k}_{prefix}_uK"] = np.nan
                return
            row[f"E_tot_{prefix}_uK"] = np.mean(Et) / UK
            row[f"E_kin_{prefix}_uK"] = np.mean(Ek) / UK
            row[f"E_pot_{prefix}_uK"] = np.mean(Et - Ek) / UK

        stats("R", Etot, Ekin)
        if len(b):
            stats("B", Eb_tot[eb], Eb_kin[eb])
        else:
            stats("B", np.empty(0), np.empty(0))

        # ------------------------------------------------------------ virial, temperatures
        if len(B) > 2:
            Kp_B = 0.5 * m * (st.vx[B] ** 2 + st.vy[B] ** 2)
            dx = st.x[B] - T.x_center(st.z[B])
            dy = st.y[B]
            W = -(dx * st.ax[B] + dy * st.ay[B]) * m          # (r - r0) . grad V
            row["virial_uK"] = np.mean(2 * Kp_B - W) / UK
            row["virial_ratio"] = np.mean(2 * Kp_B) / np.mean(W)
            vx, vy = st.vx[B], st.vy[B]
            Tp = m * (vx.var() + vy.var()) / (2 * KB)
            row["T_perp_B_uK"] = Tp * 1e6
            sB = s_b[eb]
            E_n = T.tab.energy(sB, st.n[B])
            E_0 = T.tab.energy(sB, np.zeros(len(B), np.int64))
            eps = T.ER * (E_n - E_0)
            row["T_par_B_uK"] = thermal_band_temperature(self.eps_ref, eps.mean()) * 1e6
            row["T_kin_B_uK"] = 2 * np.mean(Eb_kin[eb]) / (3 * KB) * 1e6
            row["eta_B"] = self.Ulat_c / (KB * Tp) if Tp > 0 else np.nan
            row["mean_band_B"] = st.n[B].mean()
        else:
            for k in ("virial_uK", "virial_ratio", "T_perp_B_uK", "T_par_B_uK", "T_kin_B_uK",
                      "eta_B", "mean_band_B"):
                row[k] = np.nan
        R = np.concatenate([b, f])
        if len(R) > 2:
            row["T_perp_R_uK"] = m * (st.vx[R].var() + st.vy[R].var()) / (2 * KB) * 1e6
        else:
            row["T_perp_R_uK"] = np.nan

        # ------------------------------------------------------------ collisions
        row["P_coll_ge1"] = np.mean(st.n_coll > 0)
        row["mean_ncoll_R"] = st.n_coll[R].mean() if len(R) else np.nan
        if interval and len(R):
            row["gamma_el_per_s"] = 2.0 * n_coll_interval / (len(R) * interval)
        else:
            row["gamma_el_per_s"] = np.nan

        # ------------------------------------------------------------ conservation check
        if E0 is not None and len(R):
            Eall = np.full(N, np.nan)
            Eall[R] = Etot + T.site_reference_energy(st.j[R])
            sel = R[st.n_coll[R] == 0]
            d = np.abs(Eall[sel] - E0[sel]) / self.kT0
            row["max_dE_nocoll_kT"] = float(np.nanmax(d)) if len(sel) else np.nan
        else:
            row["max_dE_nocoll_kT"] = np.nan

        # band populations of BOUND atoms (fraction of N)
        pops = np.bincount(st.n[b].astype(np.int64), minlength=self.n_band_max)[:self.n_band_max] / N
        return row, pops
