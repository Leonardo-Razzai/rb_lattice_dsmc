"""Band-resolved DSMC collisions (Bird's no-time-counter scheme) for BOUND atoms.

Cells: a cell groups K consecutive lattice sites (cell_slab_um) and a square of side h
(cell_xy_um) in the transverse plane, measured from the lattice centre line.  Only atoms
in the same site really collide; neighbouring sites have the same distribution, so pairing
test particles across the K sites is statistically equivalent provided the pair rate is
divided by K (effective cell "volume" K h^2).

Per cell with N_c test particles, each of weight N_eff:
    M_c = floor( 1/2 N_c (N_c - 1) N_eff C W_maj dt / (K h^2) + U )   candidate pairs,
accepted with probability W / W_maj, where W = sum over open final channels of |I|^2 at
the pair's mean depth and W_maj bounds W in that cell (from the overlap table).

Accepted pair: final bands (f1, f2) drawn with probability |I_{n1 n2 f1 f2}|^2 / W; the
radial centre-of-mass velocity is kept; the relative radial speed follows from energy
conservation (each atom's band energy evaluated at its own position) and its direction is
uniform in the plane (s-wave).  Atoms ending in an unbound band are released (FREE).
"""
from __future__ import annotations

import numpy as np

from ..physics.scattering import OverlapTable, pair_rate_constant
from ..state import ParticleState, Status
from . import transitions as tr


class Collider:
    def __init__(self, trap, cfg, rng, N_eff: float, dt: float, cache_dir="results/cache",
                 overlaps: OverlapTable | None = None):
        cc = cfg.collisions
        self.trap = trap
        self.rng = rng
        self.N_eff = float(N_eff)
        self.dt = float(dt)
        self.tab = overlaps or OverlapTable.load_or_compute(
            cache_dir, verbose=True, n_max=min(cc.n_bands, trap.tab.n_bands),
            s_min=cc.table_s_min, s_max=cc.table_s_max, n_s=cc.table_n_s)
        self.nmc = self.tab.n_max
        self.C = pair_rate_constant(cfg.optics.wavelength)
        self.K = max(1, int(round(cc.cell_slab_um * 1e-6 / trap.d)))
        self.h = cc.cell_xy_um * 1e-6
        z_top = trap.z_c + trap.z_half
        self.R = float(trap.rho_region(z_top)) + self.h
        self.nx = int(np.ceil(2 * self.R / self.h)) + 1
        # cumulative majorant: Mcum[k, n] bounds W for s <= s_grid[k] and bands <= n
        sg = self.tab.s_grid
        self.Mcum = np.array([self.tab.majorant_by_band(sg[min(k, len(sg) - 1)])
                              for k in range(len(sg))])
        self.mu = trap.m / 2.0
        self.reset_stats()
        self.interval = {"collisions": 0, "band_changing": 0, "released": 0}

    def reset_stats(self):
        self.stats = {"candidates": 0, "accepted": 0, "violations": 0, "conflicts": 0,
                      "band_changing": 0, "released": 0, "steps": 0}

    def pop_interval(self):
        out = dict(self.interval)
        for k in self.interval:
            self.interval[k] = 0
        return out

    # ------------------------------------------------------------------ cells
    def cell_keys(self, st, b):
        T = self.trap
        dx = st.x[b] - T.x_center(st.z[b])
        ix = np.clip(((dx + self.R) / self.h).astype(np.int64), 0, self.nx - 1)
        iy = np.clip(((st.y[b] + self.R) / self.h).astype(np.int64), 0, self.nx - 1)
        G = st.j[b] // self.K
        return (G * self.nx + ix) * self.nx + iy

    def band_energies(self, s, n_max):
        """E_R E_f(s) for f < n_max: shape (P, n_max), in J."""
        P = len(s)
        f = np.tile(np.arange(n_max), P)
        return self.trap.ER * self.trap.tab.energy(np.repeat(s, n_max), f).reshape(P, n_max)

    # ------------------------------------------------------------------ one step
    def step(self, st: ParticleState, t: float) -> int:
        rng, T = self.rng, self.trap
        self.stats["steps"] += 1
        b = np.nonzero(st.status == Status.BOUND)[0]
        if len(b) < 2:
            return 0
        if np.any(st.n[b] >= self.nmc):
            raise RuntimeError("bound atom in a band beyond the collision table; raise collisions.n_bands")
        key = self.cell_keys(st, b)
        uniq, cid = np.unique(key, return_inverse=True)
        counts = np.bincount(cid)
        order = np.argsort(cid, kind="stable")
        starts = np.cumsum(counts) - counts
        bs = b[order]                                    # state indices sorted by cell
        s_max = np.maximum.reduceat(st.s[bs], starts)
        n_max = np.maximum.reduceat(st.n[bs].astype(np.int64), starts)
        cells = np.nonzero(counts >= 2)[0]
        if len(cells) == 0:
            return 0
        k = np.clip(np.searchsorted(self.tab.s_grid, s_max[cells]), 0, len(self.tab.s_grid) - 1)
        W_maj = self.Mcum[k, n_max[cells]]
        n_c = counts[cells]
        M_exp = 0.5 * n_c * (n_c - 1) * self.N_eff * self.C * W_maj * self.dt / (self.K * self.h ** 2)
        M = np.floor(M_exp + rng.random(len(cells))).astype(np.int64)
        tot = int(M.sum())
        if tot == 0:
            return 0
        self.stats["candidates"] += tot
        rep = np.repeat(np.arange(len(cells)), M)
        nl = n_c[rep]
        il = np.floor(rng.random(tot) * nl).astype(np.int64)
        jl = np.floor(rng.random(tot) * (nl - 1)).astype(np.int64)
        jl += (jl >= il)
        st_off = starts[cells][rep]
        i = bs[st_off + il]
        j = bs[st_off + jl]
        Wm = W_maj[rep]

        # open channels and their weights
        n1, n2 = st.n[i].astype(np.int64), st.n[j].astype(np.int64)
        s1, s2 = st.s[i], st.s[j]
        E1 = self.band_energies(s1, self.nmc)
        E2 = self.band_energies(s2, self.nmc)
        ar = np.arange(tot)
        Erel = 0.25 * T.m * ((st.vx[i] - st.vx[j]) ** 2 + (st.vy[i] - st.vy[j]) ** 2)
        Etot = Erel + E1[ar, n1] + E2[ar, n2]
        open_ = (E1[:, :, None] + E2[:, None, :]) <= Etot[:, None, None]
        w = self.tab.I2_pairs(0.5 * (s1 + s2), n1, n2) * open_
        W = w.sum(axis=(1, 2))
        ratio = W / Wm
        viol = ratio > 1.0
        if viol.any():
            self.stats["violations"] += int(viol.sum())
        acc = rng.random(tot) < ratio
        if not acc.any():
            return 0
        i, j, n1, n2, Etot, w, W = i[acc], j[acc], n1[acc], n2[acc], Etot[acc], w[acc], W[acc]
        E1, E2 = E1[acc], E2[acc]
        # an atom may be claimed by several pairs in one step: keep its first pair only
        flat = np.concatenate([i, j])
        _, first = np.unique(flat, return_index=True)
        ok = np.zeros(len(flat), bool)
        ok[first] = True
        good = ok[:len(i)] & ok[len(i):]
        self.stats["conflicts"] += int((~good).sum())
        i, j, n1, n2, Etot, w, W, E1, E2 = (a[good] for a in (i, j, n1, n2, Etot, w, W, E1, E2))
        P = len(i)
        if P == 0:
            return 0
        # final channel
        cum = np.cumsum(w.reshape(P, -1), axis=1)
        u = rng.random(P) * W
        idx = np.minimum((cum < u[:, None]).sum(axis=1), self.nmc ** 2 - 1)
        f1, f2 = idx // self.nmc, idx % self.nmc
        ar = np.arange(P)
        Erel_new = np.maximum(Etot - E1[ar, f1] - E2[ar, f2], 0.0)
        g_new = np.sqrt(4.0 * Erel_new / T.m)              # 1/2 mu g^2 with mu = m/2
        vcx = 0.5 * (st.vx[i] + st.vx[j])
        vcy = 0.5 * (st.vy[i] + st.vy[j])
        ph = 2.0 * np.pi * rng.random(P)
        cx, cy = np.cos(ph), np.sin(ph)
        st.vx[i], st.vy[i] = vcx + 0.5 * g_new * cx, vcy + 0.5 * g_new * cy
        st.vx[j], st.vy[j] = vcx - 0.5 * g_new * cx, vcy - 0.5 * g_new * cy
        changed = (np.minimum(f1, f2) != np.minimum(n1, n2)) | (np.maximum(f1, f2) != np.maximum(n1, n2))
        st.n[i], st.n[j] = f1, f2
        st.n_coll[i] += 1
        st.n_coll[j] += 1
        self.stats["accepted"] += P
        self.stats["band_changing"] += int(changed.sum())
        self.interval["collisions"] += P
        self.interval["band_changing"] += int(changed.sum())

        # release atoms that ended in an unbound band; refresh forces of the others
        both = np.concatenate([i, j])
        fb = np.concatenate([f1, f2]).astype(np.int64)
        sb = st.s[both]
        unb = ~T.band_is_bound(sb, fb)
        if unb.any():
            ui = both[unb]
            E_band = T.ER * T.tab.energy(sb[unb], fb[unb])
            tr.release_to_free(T, st, ui, E_band, sb[unb] * T.ER, st.z[ui], t, rng)
            self.stats["released"] += int(unb.sum())
            self.interval["released"] += int(unb.sum())
        keep = both[~unb]
        if len(keep):
            e = T.bound_eval(st.x[keep], st.y[keep], st.j[keep], st.n[keep], z_guess=st.z[keep])
            st.ax[keep], st.ay[keep] = (e.F / T.m).T
        return P
